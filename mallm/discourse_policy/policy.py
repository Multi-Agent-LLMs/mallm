from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.utils.types import Agreement, Memory, TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
logger = logging.getLogger("mallm")


class DiscoursePolicy(ABC):
    def __init__(self, paradigm_str: str = "") -> None:
        self.paradigm_str = paradigm_str
        self.decision = False
        self.turn = 0
        self.unique_id = 0
        self.memories: list[Memory] = []
        self.draft = ""
        self.agreements: list[Agreement] = []

    def discuss(
        self,
        coordinator: Coordinator,
        task_instruction: str,
        input_str: str,
        use_moderator: bool = False,
        feedback_sentences: Optional[tuple[int, int]] = None,
        max_turns: int = 10,
        force_all_turns: bool = False,
        context_length: int = 1,
        include_current_turn_in_memory: bool = False,
        debate_rounds: int = 1,
        chain_of_thought: bool = True,
    ) -> tuple[Optional[str], int, list[Agreement]]:
        logger.debug(self.paradigm_str)
        while (not self.decision or force_all_turns) and self.turn < max_turns:
            self.turn += 1
            logger.info(f"Ongoing. Current turn: {self.turn}")

            for i, agent in enumerate(coordinator.agents):
                debate_history, memory_ids, current_draft = (
                    agent.get_discussion_history(
                        context_length=context_length,
                        turn=self.turn,
                        include_this_turn=include_current_turn_in_memory,
                    )
                )

                template_filling = TemplateFilling(
                    task_instruction=task_instruction,
                    input_str=input_str,
                    current_draft=current_draft,
                    persona=agent.persona,
                    persona_description=agent.persona_description,
                    agent_memory=debate_history,
                    feedback_sentences=feedback_sentences,
                )

                if isinstance(agent, Moderator):
                    template_filling.feedback_sentences = None
                    self.moderator_call(
                        moderator=agent,
                        template_filling=template_filling,
                        memory_ids=memory_ids,
                        agent_index=i,
                        coordinator=coordinator,
                        chain_of_thought=chain_of_thought,
                    )
                elif isinstance(agent, Panelist):
                    self.panelist_call(
                        agent_index=i,
                        template_filling=template_filling,
                        coordinator=coordinator,
                        memory_ids=memory_ids,
                        agent=agent,
                        chain_of_thought=chain_of_thought,
                    )
                else:
                    logger.error("Agent type not recognized.")
                    raise Exception("Agent type not recognized.")
                self.unique_id += 1
                self.memories = []

                if coordinator.decision_making is None:
                    logger.error("No decision making module found.")
                    raise Exception("No decision making module found.")

                self.draft, self.decision = coordinator.decision_making.make_decision(
                    self.agreements, self.turn, task_instruction, input_str
                )
                if self.decision:
                    break

        return current_draft, self.turn, self.agreements

    @abstractmethod
    def moderator_call(
        self,
        moderator: Moderator,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        chain_of_thought: bool,
    ) -> None:
        pass

    @abstractmethod
    def panelist_call(
        self,
        agent: Panelist,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        chain_of_thought: bool,
    ) -> None:
        pass

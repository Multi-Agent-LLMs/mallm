from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.utils.types import Agreement, TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator

logger = logging.getLogger("mallm")


class DiscourseDebate(DiscoursePolicy):
    def moderator_call(
        self,
        moderator: Moderator,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        pass

    def panelist_call(
        self,
        agent: Panelist,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        pass

    def __init__(self, debate_rounds: int = 1):
        super().__init__("")
        self.debate_rounds = debate_rounds

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
        debate_rounds: int = 2,
    ) -> tuple[Optional[str], int, list[Agreement], bool]:
        unique_id = 0
        memories = []

        logger.debug(
            f"""Paradigm: Debate (rounds: {debate_rounds})
                            ┌───┐
                  ┌────────►│A 1│◄────────┐
                  │         └───┘         │
                  │                       │
                  │                       │
                  │                       │
                ┌─┴─┬──────────────────►┌─┴─┐
                │A 3│                   │A 2│
                └───┘◄──────────────────┴───┘
                """
        )

        logger.info("Debate rounds between agents A2, ..., An: " + str(debate_rounds))

        while (not self.decision or force_all_turns) and self.turn < max_turns:
            self.turn += 1
            logger.info("Ongoing. Current turn: " + str(self.turn))

            # ---- Agent A1
            if use_moderator and coordinator.moderator is not None:
                debate_history, memory_ids, current_draft = (
                    coordinator.moderator.get_discussion_history(
                        context_length=context_length,
                        turn=self.turn,
                        include_this_turn=include_current_turn_in_memory,
                    )
                )
                template_filling = TemplateFilling(
                    task_instruction=task_instruction,
                    input_str=input_str,
                    current_draft=current_draft,
                    persona=coordinator.moderator.persona,
                    persona_description=coordinator.moderator.persona_description,
                    agent_memory=debate_history,
                )
                _res, memory, self.agreements = coordinator.moderator.draft(
                    unique_id=unique_id,
                    turn=self.turn,
                    memory_ids=memory_ids,
                    template_filling=template_filling,
                    agreements=self.agreements,
                    is_moderator=True,
                )
                memories.append(memory)
                coordinator.update_memories(memories, coordinator.agents)
                memories = []
                unique_id += 1
            else:
                debate_history, memory_ids, current_draft = coordinator.panelists[
                    0
                ].get_discussion_history(
                    context_length=context_length,
                    turn=self.turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                template_filling = TemplateFilling(
                    task_instruction=task_instruction,
                    input_str=input_str,
                    current_draft=current_draft,
                    persona=coordinator.panelists[0].persona,
                    persona_description=coordinator.panelists[0].persona_description,
                    agent_memory=debate_history,
                )
                _res, memory, self.agreements = coordinator.panelists[0].draft(
                    unique_id=unique_id,
                    turn=self.turn,
                    memory_ids=memory_ids,
                    template_filling=template_filling,
                    agreements=self.agreements,
                    is_moderator=True,
                )
                memories.append(memory)
                coordinator.update_memories(memories, coordinator.agents)
                memories = []
                unique_id += 1

            for r in range(debate_rounds):  # ---- Agents A2, A3, ...
                logger.debug("Debate round: " + str(r))
                debate_agreements: list[Agreement] = []
                for i, a in enumerate(
                    coordinator.agents[1:]
                ):  # similar to relay paradigm
                    # Because we should only iterate over Panelists with [1:]
                    # We call participate() below, which is a method of Panelist
                    assert isinstance(a, Panelist)

                    debate_history, memory_ids, current_draft = (
                        a.get_discussion_history(
                            context_length=context_length,
                            turn=self.turn,
                            include_this_turn=include_current_turn_in_memory,
                        )
                    )
                    next_a = i + 2
                    if i == len(coordinator.agents[1:]) - 1:
                        next_a = 1  # start again with agent 1 (loop)

                    template_filling = TemplateFilling(
                        task_instruction=task_instruction,
                        input_str=input_str,
                        current_draft=current_draft,
                        persona=a.persona,
                        persona_description=a.persona_description,
                        agent_memory=debate_history,
                        feedback_sentences=feedback_sentences,
                    )

                    if r == debate_rounds - 1:  # last debate round
                        agents_to_update = [
                            coordinator.agents[0],
                            a,
                            coordinator.agents[next_a],
                        ]
                    else:
                        agents_to_update = [a, coordinator.agents[next_a]]
                    debate_agreements = a.participate(
                        use_moderator=True,  # only feedback makes sense with the debate paradigm
                        memories=memories,
                        unique_id=unique_id,
                        turn=self.turn,
                        memory_ids=memory_ids,
                        template_filling=template_filling,
                        agents_to_update=agents_to_update,
                        agreements=debate_agreements,
                    )
                    if len(debate_agreements) > len(coordinator.agents) - 1:
                        debate_agreements = debate_agreements[
                            -(len(coordinator.agents) - 1) :
                        ]
                    unique_id += 1

            self.agreements += debate_agreements

            if coordinator.decision_protocol is None:
                logger.error("No decision protocol module found.")
                raise Exception("No decision protocol module found.")

            self.draft, self.decision, self.agreements = (
                coordinator.decision_protocol.make_decision(
                    self.agreements,
                    self.turn,
                    len(coordinator.agents),
                    task_instruction,
                    input_str,
                )
            )
            if self.decision:
                break

        return self.draft, self.turn, self.agreements, self.decision

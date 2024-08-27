from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from rich.progress import Console  # type: ignore

from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.utils.types import Agreement, TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
    from mallm.utils.config import Config

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
        config: Config,
        console: Optional[Console] = None,
    ) -> tuple[Optional[str], int, list[Agreement], bool]:
        unique_id = 0
        memories = []
        voting_process_string = ""
        if console is None:
            console = Console()

        logger.info(
            f"""Paradigm: Debate (rounds: {config.debate_rounds})
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

        logger.info(
            "Debate rounds between agents A2, ..., An: " + str(config.debate_rounds)
        )

        while (
            not self.decision or config.force_all_turns
        ) and self.turn < config.max_turns:
            self.turn += 1
            logger.info("Ongoing. Current turn: " + str(self.turn))

            # ---- Agent A1
            if config.use_moderator and coordinator.moderator is not None:
                discussion_history, memory_ids, current_draft = (
                    coordinator.moderator.get_discussion_history(
                        context_length=config.context_length,
                        turn=self.turn,
                        include_this_turn=config.include_current_turn_in_memory,
                    )
                )
                if self.turn == 1 and config.all_agents_generate_first_draft:
                    current_draft = None
                    discussion_history = None
                template_filling = TemplateFilling(
                    task_instruction=task_instruction,
                    input_str=input_str,
                    current_draft=current_draft,
                    persona=coordinator.moderator.persona,
                    persona_description=coordinator.moderator.persona_description,
                    agent_memory=discussion_history,
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
                discussion_history, memory_ids, current_draft = coordinator.panelists[
                    0
                ].get_discussion_history(
                    context_length=config.context_length,
                    turn=self.turn,
                    include_this_turn=config.include_current_turn_in_memory,
                )
                if self.turn == 1 and config.all_agents_generate_first_draft:
                    current_draft = None
                    discussion_history = None
                template_filling = TemplateFilling(
                    task_instruction=task_instruction,
                    input_str=input_str,
                    current_draft=current_draft,
                    persona=coordinator.panelists[0].persona,
                    persona_description=coordinator.panelists[0].persona_description,
                    agent_memory=discussion_history,
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

            for r in range(config.debate_rounds):  # ---- Agents A2, A3, ...
                logger.debug("Debate round: " + str(r))
                debate_agreements: list[Agreement] = []
                for i, a in enumerate(
                    coordinator.agents[1:]
                ):  # similar to relay paradigm
                    # Because we should only iterate over Panelists with [1:]
                    # We call participate() below, which is a method of Panelist
                    assert isinstance(a, Panelist)

                    discussion_history, memory_ids, current_draft = (
                        a.get_discussion_history(
                            context_length=config.context_length,
                            turn=self.turn,
                            include_this_turn=config.include_current_turn_in_memory,
                        )
                    )
                    next_a = i + 2
                    if i == len(coordinator.agents[1:]) - 1:
                        next_a = 1  # start again with agent 1 (loop)
                    if (
                        self.turn == 1
                        and r == 1
                        and config.all_agents_generate_first_draft
                    ):
                        current_draft = None
                        discussion_history = None
                    template_filling = TemplateFilling(
                        task_instruction=task_instruction,
                        input_str=input_str,
                        current_draft=current_draft,
                        persona=a.persona,
                        persona_description=a.persona_description,
                        agent_memory=discussion_history,
                        feedback_sentences=config.feedback_sentences,
                    )

                    if r == config.debate_rounds - 1:  # last debate round
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
                            1 - len(coordinator.agents) :
                        ]
                    unique_id += 1

            self.agreements += debate_agreements

            if coordinator.decision_protocol is None:
                logger.error("No decision protocol module found.")
                raise Exception("No decision protocol module found.")

            self.draft, self.decision, self.agreements, voting_process_string = (
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
            self.print_messages(coordinator, input_str, task_instruction)
        self.print_messages(
            coordinator,
            input_str,
            task_instruction,
            False,
            voting_process_string,
            console=console,
        )
        return self.draft, self.turn, self.agreements, self.decision

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from rich.progress import Console  # type: ignore

from mallm.agents.draftProposer import DraftProposer
from mallm.agents.panelist import Panelist
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.utils.types import Agreement, TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator

logger = logging.getLogger("mallm")


class DiscourseDebate(DiscoursePolicy):
    def draft_proposer_call(
        self,
        draft_proposer: DraftProposer,
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
        num_neutral_agents: int = 0,
        feedback_sentences: Optional[tuple[int, int]] = None,
        max_turns: int = 10,
        force_all_turns: bool = False,
        context_length: int = 1,
        include_current_turn_in_memory: bool = False,
        debate_rounds: int = 2,
        console: Optional[Console] = None,
    ) -> tuple[Optional[str], int, list[Agreement], bool]:
        unique_id = 0
        memories = []
        voting_process_string = ""
        if console is None:
            console = Console()

        logger.info(
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
            debate_history, memory_ids, current_draft = (
                coordinator.agents[0].get_discussion_history(
                    context_length=context_length,
                    turn=self.turn,
                    include_this_turn=include_current_turn_in_memory,
                )
            )
            template_filling = TemplateFilling(
                task_instruction=task_instruction,
                input_str=input_str,
                current_draft=current_draft,
                persona=coordinator.agents[0].persona,
                persona_description=coordinator.agents[0].persona_description,
                agent_memory=debate_history,
            )
            _res, memory, self.agreements = coordinator.agents[0].draft(
                unique_id=unique_id,
                turn=self.turn,
                memory_ids=memory_ids,
                template_filling=template_filling,
                agreements=self.agreements,
                is_neutral=True,
            )
            memories.append(memory)
            coordinator.update_memories(memories, coordinator.agents)
            memories = []
            unique_id += 1

            # ---- Agents A2, A3, ...
            for r in range(debate_rounds):
                logger.debug(f"Discussion {coordinator.id} goes into debate round: {r!s}")
                debate_agreements: list[Agreement] = []
                for i, a in enumerate(
                    coordinator.agents[1:]
                ):  # similar to relay paradigm
                    # Because we should only iterate over Panelists with [1:]
                    # We call participate() below, which is a method of Panelist

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

                    if isinstance(a, DraftProposer):
                        _res, _debate_memory, debate_agreements = a.draft(
                            unique_id=unique_id,
                            turn=self.turn,
                            memory_ids=memory_ids,
                            template_filling=template_filling,
                            agreements=debate_agreements,
                            is_neutral=True,
                        )
                    else:
                        debate_agreements = a.participate(
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

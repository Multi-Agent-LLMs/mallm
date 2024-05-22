from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from mallm.agents.panelist import Panelist
from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy
from mallm.utils.types import Agreement, TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator

logger = logging.getLogger("mallm")


class DiscourseDebate(DiscoursePolicy):
    def discuss(
        self,
        coordinator: Coordinator,
        task_instruction: str,
        input_str: str,
        use_moderator: bool = False,
        feedback_sentences: tuple[int, int] = (3, 4),
        max_turns: Optional[int] = None,
        context_length: int = 1,
        include_current_turn_in_memory: bool = False,
        extract_all_drafts: bool = False,
        debate_rounds: int = 1,
    ):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements: list[Agreement] = []

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

        while not decision and (max_turns is None or turn < max_turns):
            turn = turn + 1
            logger.info("Ongoing. Current turn: " + str(turn))

            # ---- Agent A1
            if use_moderator and coordinator.moderator is not None:
                debate_history, memory_ids, current_draft = (
                    coordinator.moderator.get_debate_history(
                        context_length=context_length,
                        turn=turn,
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
                res, memory, agreements = coordinator.moderator.draft(
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    agreements,
                    is_moderator=True,
                )
                memories.append(memory)
                memories = coordinator.update_memories(memories, coordinator.agents)
                unique_id = unique_id + 1
            else:
                debate_history, memory_ids, current_draft = coordinator.panelists[
                    0
                ].get_debate_history(
                    context_length=context_length,
                    turn=turn,
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
                res, memory, agreements = coordinator.panelists[0].draft(
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    agreements,
                    is_moderator=True,
                )
                memories.append(memory)
                memories = coordinator.update_memories(memories, coordinator.agents)
                unique_id = unique_id + 1

            for r in range(debate_rounds):  # ---- Agents A2, A3, ...
                logger.debug("Debate round: " + str(r))
                debate_agreements: list[Agreement] = []
                for i, a in enumerate(
                    coordinator.agents[1:]
                ):  # similar to relay paradigm
                    # Because we should only iterate over Panelists with [1:]
                    # We call participate() below, which is a method of Panelist
                    assert isinstance(a, Panelist)

                    debate_history, memory_ids, current_draft = a.get_debate_history(
                        context_length=context_length,
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory,
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
                        sents_max=feedback_sentences[1],
                        sents_min=feedback_sentences[0],
                    )

                    if r == debate_rounds - 1:  # last debate round
                        agents_to_update = [
                            coordinator.agents[0],
                            a,
                            coordinator.agents[next_a],
                        ]
                    else:
                        agents_to_update = [a, coordinator.agents[next_a]]
                    memories, debate_agreements = a.participate(
                        use_moderator,
                        memories,
                        unique_id,
                        turn,
                        memory_ids,
                        template_filling,
                        extract_all_drafts,
                        agents_to_update,
                        debate_agreements,
                    )
                    if len(debate_agreements) > len(coordinator.agents) - 1:
                        debate_agreements = debate_agreements[
                            1 - len(coordinator.agents) :
                        ]
                    unique_id = unique_id + 1

            agreements = agreements + debate_agreements
            if len(agreements) > len(coordinator.panelists):
                agreements = agreements[-len(coordinator.panelists) :]

            if coordinator.decision_making is None:
                logger.error("No decision making module found.")
                raise Exception("No decision making module found.")

            draft, decision = coordinator.decision_making.make_decision(
                agreements, turn, task_instruction, input_str
            )

        return draft, turn, agreements

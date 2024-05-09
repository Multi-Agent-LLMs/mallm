from __future__ import annotations

import logging

from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy
from typing import TYPE_CHECKING

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
        feedback_sentences: list[int] = (3, 4),
        max_turns: int = None,
        context_length: int = 1,
        include_current_turn_in_memory: bool = False,
        extract_all_drafts: bool = False,
        debate_rounds: int = 1,
    ):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

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

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            log = "Ongoing. Current turn: " + str(turn)
            logger.info("Ongoing. Current turn: " + str(turn))

            # ---- Agent A1
            if use_moderator:
                memory_string, memory_ids, current_draft = (
                    coordinator.moderator.get_memory_string(
                        context_length=context_length,
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory,
                    )
                )

                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input_str,
                    "currentDraft": current_draft,
                    "persona": coordinator.moderator.persona,
                    "personaDescription": coordinator.moderator.persona_description,
                    "agentMemory": memory_string,
                }
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
                memory_string, memory_ids, current_draft = coordinator.panelists[
                    0
                ].get_memory_string(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input_str,
                    "currentDraft": current_draft,
                    "persona": coordinator.panelists[0].persona,
                    "personaDescription": coordinator.panelists[0].persona_description,
                    "agentMemory": memory_string,
                }
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
                debate_agreements = []
                for i, a in enumerate(
                    coordinator.agents[1:]
                ):  # similar to relay paradigm
                    memory_string, memory_ids, current_draft = a.get_memory_string(
                        context_length=context_length,
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory,
                    )
                    next_a = i + 2
                    if i == len(coordinator.agents[1:]) - 1:
                        next_a = 1  # start again with agent 1 (loop)

                    template_filling = {
                        "taskInstruction": task_instruction,
                        "input": input_str,
                        "currentDraft": current_draft,
                        "persona": a.persona,
                        "personaDescription": a.persona_description,
                        "sentsMin": feedback_sentences[0],
                        "sentsMax": feedback_sentences[1],
                        "agentMemory": memory_string,
                    }
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
            decision = coordinator.decision_making.decide(agreements, turn)

        return current_draft, turn, agreements

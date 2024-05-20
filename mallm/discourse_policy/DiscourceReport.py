from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
logger = logging.getLogger("mallm")


class DiscourseReport(DiscoursePolicy):
    def discuss(
        self,
        coordinator: Coordinator,
        task_instruction: str,
        input_str: str,
        use_moderator: bool = False,
        feedback_sentences: tuple[int, int] = (3, 4),
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
            """Paradigm: Report
                    ┌───┐
                    │A 1│
            ┌──────►└┼─┼┘◄──────┐
            │        │ │        │
            │        │ │        │
            │        │ │        │
        ┌───┼◄───────┘ └───────►├───┐
        │A 3│                   │A 2│
        └───┘                   └───┘
        """
        )

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            logger.info("Ongoing. Current turn: " + str(turn))

            # ---- Agent A1
            if use_moderator:
                debate_history, memory_ids, current_draft = (
                    coordinator.moderator.get_debate_history(
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
                    "agentMemory": debate_history,
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
                debate_history, memory_ids, current_draft = coordinator.panelists[
                    0
                ].get_debate_history(
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
                    "agentMemory": debate_history,
                }
                res, memory, agreements = coordinator.panelists[0].draft(
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    agreements,
                )
                memories.append(memory)
                memories = coordinator.update_memories(memories, coordinator.agents)
                unique_id = unique_id + 1

            # ---- Agents A2, A3, A4, ...
            for p in coordinator.agents[1:]:
                debate_history, memory_ids, current_draft = p.get_debate_history(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input_str,
                    "currentDraft": current_draft,
                    "persona": p.persona,
                    "personaDescription": p.persona_description,
                    "sentsMin": feedback_sentences[0],
                    "sentsMax": feedback_sentences[1],
                    "agentMemory": debate_history,
                }

                memories, agreements = p.participate(
                    True,
                    memories,
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    [coordinator.agents[0], p],
                    agreements,
                )
                unique_id = unique_id + 1

            draft, decision = coordinator.decision_making.make_decision(
                agreements, turn, task_instruction, input_str
            )
        return draft, turn, agreements

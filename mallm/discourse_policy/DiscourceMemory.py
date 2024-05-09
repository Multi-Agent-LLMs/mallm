from __future__ import annotations

import logging

from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
logger = logging.getLogger("mallm")


class DiscourseMemory(DiscoursePolicy):

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
            """Paradigm: Memory
                    ┌───┐
                    │A 1│
                    ├───┘
                    │   ▲
                    │   │
                    ▼   │
        ┌───┬──────►┌───┤◄──────┬───┐
        │A 3│       │MEM│       │A 2│
        └───┘◄──────┴───┴──────►└───┘
        """
        )
        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            logger.info(
                "Discussion " + coordinator.id + " ongoing. Current turn: " + str(turn)
            )

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

            for p in coordinator.panelists:
                memory_string, memory_ids, current_draft = p.get_memory_string(
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
                    "agentMemory": memory_string,
                }

                memories, agreements = p.participate(
                    use_moderator,
                    memories,
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    coordinator.agents,
                    agreements,
                )
                unique_id = unique_id + 1

            decision = coordinator.decision_making.decide(agreements, turn)

        return current_draft, turn, agreements

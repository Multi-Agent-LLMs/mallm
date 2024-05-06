from __future__ import annotations

import logging

from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
logger = logging.getLogger("mallm")


class DiscourseRelay(DiscoursePolicy):
    def discuss(
        self,
        coordinator: Coordinator,
        task_instruction: str,
        input: str,
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
            """Paradigm: Relay
                    ┌───┐
          ┌────────►│A 1│─────────┐
          │         └───┘         │
          │                       │
          │                       │
          │                       ▼
        ┌─┴─┐                   ┌───┐
        │A 3│◄──────────────────┤A 2│
        └───┘                   └───┘
        """
        )

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            logger.info("Ongoing. Current turn: " + str(turn))

            for i, a in enumerate(coordinator.agents):
                memory_string, memory_ids, current_draft = a.get_memory_string(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                next_a = i + 1
                if i == len(coordinator.agents) - 1:
                    next_a = 0  # start again with agent 0 (loop)
                if a == coordinator.moderator:
                    template_filling = {
                        "task_instruction": task_instruction,
                        "input": input,
                        "current_draft": current_draft,
                        "persona": coordinator.moderator.persona,
                        "persona_description": coordinator.moderator.persona_description,
                        "agent_memory": memory_string,
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
                    memories = coordinator.update_memories(
                        memories, [a, coordinator.agents[next_a]]
                    )
                else:
                    template_filling = {
                        "taskInstruction": task_instruction,
                        "input": input,
                        "currentDraft": current_draft,
                        "persona": a.persona,
                        "personaDescription": a.persona_description,
                        "sentsMin": feedback_sentences[0],
                        "sentsMax": feedback_sentences[1],
                        "agentMemory": memory_string,
                    }
                    memories, agreements = a.participate(
                        use_moderator,
                        memories,
                        unique_id,
                        turn,
                        memory_ids,
                        template_filling,
                        extract_all_drafts,
                        [a, coordinator.agents[next_a]],
                        agreements,
                    )
                unique_id = unique_id + 1

            decision = coordinator.decision_making.decide(agreements, turn)

        return current_draft, turn, agreements

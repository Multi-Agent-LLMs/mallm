from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy
from mallm.utils.types import Agreement, TemplateFilling

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
        while not decision and (max_turns is None or turn < max_turns):
            turn = turn + 1
            logger.info(
                "Discussion " + coordinator.id + " ongoing. Current turn: " + str(turn)
            )

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

            for p in coordinator.panelists:
                debate_history, memory_ids, current_draft = p.get_debate_history(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )

                template_filling = TemplateFilling(
                    task_instruction=task_instruction,
                    input_str=input_str,
                    current_draft=current_draft,
                    persona=p.persona,
                    persona_description=p.persona_description,
                    agent_memory=debate_history,
                    sents_max=feedback_sentences[1],
                    sents_min=feedback_sentences[0],
                )

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

            if coordinator.decision_making is None:
                logger.error("No decision making module found.")
                raise Exception("No decision making module found.")

            draft, decision = coordinator.decision_making.make_decision(
                agreements, turn, task_instruction, input_str
            )

        return draft, turn, agreements

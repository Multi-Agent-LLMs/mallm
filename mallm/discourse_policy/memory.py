from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.utils.types import TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
logger = logging.getLogger("mallm")


class DiscourseMemory(DiscoursePolicy):
    def __init__(self) -> None:
        super().__init__(
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

    def panelist_call(
        self,
        agent: Panelist,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        self.agreements = agent.participate(
            use_moderator=coordinator.moderator is not None,
            memories=self.memories,
            unique_id=self.unique_id,
            turn=self.turn,
            memory_ids=memory_ids,
            template_filling=template_filling,
            agents_to_update=coordinator.agents,
            agreements=self.agreements,
        )

    def moderator_call(
        self,
        moderator: Moderator,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        res, memory, self.agreements = moderator.draft(
            unique_id=self.unique_id,
            turn=self.turn,
            memory_ids=memory_ids,
            template_filling=template_filling,
            agreements=self.agreements,
            is_moderator=True,
        )
        self.memories.append(memory)
        coordinator.update_memories(self.memories, coordinator.agents)
        self.memories = []

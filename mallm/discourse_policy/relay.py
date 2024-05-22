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


class DiscourseRelay(DiscoursePolicy):
    def __init__(self) -> None:
        super().__init__(
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

    def panelist_call(
        self,
        agent: Panelist,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        extract_all_drafts: bool,
    ) -> None:
        next_agent = (agent_index + 1) % (len(coordinator.agents) - 1)
        self.agreements = agent.participate(
            coordinator.moderator is not None,
            self.memories,
            self.unique_id,
            self.turn,
            memory_ids,
            template_filling,
            extract_all_drafts,
            [agent, coordinator.agents[next_agent]],
            self.agreements,
        )

    def moderator_call(
        self,
        moderator: Moderator,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        extract_all_drafts: bool,
    ) -> None:
        next_agent = (agent_index + 1) % (len(coordinator.agents) - 1)
        res, memory, self.agreements = moderator.draft(
            self.unique_id,
            self.turn,
            memory_ids,
            template_filling,
            extract_all_drafts,
            self.agreements,
            is_moderator=True,
        )
        self.memories.append(memory)
        coordinator.update_memories(
            self.memories, [moderator, coordinator.agents[next_agent]]
        )
        self.memories = []

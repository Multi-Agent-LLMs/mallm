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
        chain_of_thought: bool,
    ) -> None:
        next_agent = (agent_index + 1) % len(coordinator.agents)
        self.agreements = agent.participate(
            use_moderator=coordinator.moderator is not None,
            memories=self.memories,
            unique_id=self.unique_id,
            turn=self.turn,
            memory_ids=memory_ids,
            template_filling=template_filling,
            extract_all_drafts=extract_all_drafts,
            agents_to_update=[agent, coordinator.agents[next_agent]],
            agreements=self.agreements,
            chain_of_thought=chain_of_thought,
        )

    def moderator_call(
        self,
        moderator: Moderator,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        extract_all_drafts: bool,
        chain_of_thought: bool,
    ) -> None:
        next_agent = (agent_index + 1) % len(coordinator.agents)
        res, memory, self.agreements = moderator.draft(
            unique_id=self.unique_id,
            turn=self.turn,
            memory_ids=memory_ids,
            template_filling=template_filling,
            extract_all_drafts=extract_all_drafts,
            agreements=self.agreements,
            is_moderator=True,
            chain_of_thought=chain_of_thought,
        )
        self.memories.append(memory)
        coordinator.update_memories(
            self.memories, [moderator, coordinator.agents[next_agent]]
        )
        self.memories = []
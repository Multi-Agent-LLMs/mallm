from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mallm.agents.draftProposer import DraftProposer
from mallm.agents.panelist import Panelist
from mallm.discussion_paradigms.paradigm import DiscussionParadigm
from mallm.utils.types import TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
logger = logging.getLogger("mallm")


class DiscussionRelay(DiscussionParadigm):
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
    ) -> None:
        next_agent = (agent_index + 1) % len(coordinator.agents)
        self.agreements = agent.participate(
            memories=self.memories,
            unique_id=self.unique_id,
            turn=self.turn,
            memory_ids=memory_ids,
            template_filling=template_filling,
            agents_to_update=[agent, coordinator.agents[next_agent]],
            agreements=self.agreements,
        )

    def draft_proposer_call(
        self,
        draft_proposer: DraftProposer,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        next_agent = (agent_index + 1) % len(coordinator.agents)
        _res, memory, self.agreements = draft_proposer.draft(
            unique_id=self.unique_id,
            turn=self.turn,
            memory_ids=memory_ids,
            template_filling=template_filling,
            agreements=self.agreements,
            is_neutral=True,
        )
        self.memories.append(memory)
        coordinator.update_memories(
            self.memories, [draft_proposer, coordinator.agents[next_agent]]
        )
        self.memories = []

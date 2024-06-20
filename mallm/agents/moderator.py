from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from mallm.agents.agent import Agent
from mallm.models.Chat import Chat
from mallm.utils.types import Agreement

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator


class Moderator(Agent):
    def __init__(
        self,
        llm: Chat,
        client: httpx.Client,
        coordinator: Coordinator,
        persona: str = "Moderator",
        persona_description: str = "A super-intelligent individual with critical thinking who has a neutral position at all times. He acts as a mediator between other discussion participants.",
    ) -> None:
        super().__init__(llm, client, coordinator, persona, persona_description)

    def agree(
        self, res: str, agreements: list[Agreement], self_drafted: bool = True
    ) -> list[Agreement]:
        """
        Determines whether a string given by an agent means an agreement or disagreement.
        Returns a list of bools
        """
        return [
            Agreement(
                agreement=None, agent_id=self.id, persona=self.persona, response=res
            )
        ]

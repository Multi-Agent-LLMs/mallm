from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from mallm.agents.agent import Agent
from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator


class DraftProposer(Agent):
    def __init__(
        self,
        llm: Chat,
        client: httpx.Client,
        coordinator: Coordinator,
        response_generator: ResponseGenerator,
        persona: str = "Moderator",
        persona_description: str = "A super-intelligent individual with critical thinking who has a neutral position at all times. He acts as a mediator between other discussion participants.",
    ) -> None:
        super().__init__(
            llm, client, coordinator, response_generator, persona, persona_description
        )

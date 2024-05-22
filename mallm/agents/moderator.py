from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from langchain_core.language_models import LLM

from mallm.agents.agent import Agent

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator


class Moderator(Agent):
    def __init__(
        self,
        llm: LLM,
        client: httpx.Client,
        coordinator: Coordinator,
        persona: str = "Moderator",
        persona_description: str = "A super-intelligent individual with critical thinking who has a neutral position at all times. He acts as a mediator between other discussion participants.",
    ) -> None:
        super().__init__(llm, client, coordinator, persona, persona_description)

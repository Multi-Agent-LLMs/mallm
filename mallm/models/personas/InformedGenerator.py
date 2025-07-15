import logging
from typing import Optional

from mallm.models.Chat import Chat
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.utils.types import InputExample

logger = logging.getLogger("mallm")


class InformedGenerator(PersonaGenerator):
    """
    The InformedPersona class is a specialized PersonaGenerator designed to generate personas with unique knowledge. The personas take the information for their system prompt.
    """

    def __init__(self, llm: Chat):
        self.llm = llm

    @classmethod
    def generate_persona(
        cls,
        task_description: str,
        already_generated_personas: list[dict[str, str]],
        sample: InputExample,
    ) -> dict[str, str]:
        return {
            "role": f"Participant {len(already_generated_personas) + 1}",
            "description": "A participant of the discussion.",
        }

    @classmethod
    def generate_persona(
        cls,
        task_description: str,
        already_generated_personas: list[dict[str, str]],
        sample: InputExample,
        informations: list[Optional[str]],
    ) -> dict[str, str]:
        if informations[len(already_generated_personas)] is not None:
            return {
                "role": f"Participant {len(already_generated_personas) + 1}",
                "description": f"A participant of the discussion with the following information: {informations[len(already_generated_personas)]}",
            }
        return {
            "role": f"Participant {len(already_generated_personas) + 1}",
            "description": "A participant of the discussion.",
        }

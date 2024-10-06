import logging

from mallm.models.Chat import Chat
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.utils.types import InputExample

logger = logging.getLogger("mallm")


class NoPersonaGenerator(PersonaGenerator):
    """
    The NoPersonaGenerator class is used to create agents without a persona.
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

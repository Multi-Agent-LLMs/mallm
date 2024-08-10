from abc import ABC, abstractmethod
from typing import Any

from mallm.models.Chat import Chat
from mallm.utils.types import InputExample


class PersonaGenerator(ABC):
    @abstractmethod
    def __init__(self, llm: Chat):
        pass

    @abstractmethod
    def generate_personas(
        self, task_description: str, num_agents: int, sample: InputExample
    ) -> list[dict[str, str]]:
        """
        Abstract method to generate a list of persona descriptions based on the given task.

        Parameters:
        task_description (str): Description of the task for which personas are to be generated.
        num_agents (int): Number of persona descriptions to generate.

        Returns:
        list[dict[str, str]]: A list of dictionaries, where each dictionary represents a persona
                               with keys like 'role' and 'persona' (or other relevant descriptors),
                               each mapped to their respective string description.
        """

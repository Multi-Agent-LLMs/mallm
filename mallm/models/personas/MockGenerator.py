from mallm.models.Chat import Chat
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.utils.types import InputExample


class MockGenerator(PersonaGenerator):
    def __init__(self, llm: Chat):
        pass

    @staticmethod
    def generate_personas(
        task_description: str, num_agents: int, sample: InputExample
    ) -> list[dict[str, str]]:
        return [
            {"role": f"Panelist {i}", "description": "generic"}
            for i in range(1, num_agents + 1)
        ]

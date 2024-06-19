from mallm.models.Chat import Chat
from mallm.models.personas.PersonaGenerator import PersonaGenerator


class MockGenerator(PersonaGenerator):
    def __init__(self, llm: Chat):
        pass

    def generate_personas(
        self, task_description: str, num_agents: int
    ) -> list[dict[str, str]]:
        return [
            {"role": f"Panelist {i}", "description": "generic"}
            for i in range(1, num_agents + 1)
        ]

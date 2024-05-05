import json
from huggingface_hub import InferenceClient

from mallm.models.personas.PersonaGenerator import PersonaGenerator


class TGIPersonaGenerator(PersonaGenerator):
    def __init__(self, endpoint_url):
        self.client = InferenceClient(endpoint_url)
        self.persona_grammar = {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "persona": {"type": "string"},
            },
            "required": ["role", "persona"],
        }
        self.base_prompt = """
        Generate a personas using the JSON schema provided. Here are some example tasks and their corresponding personas:

        Task: Explain the basics of machine learning to high school students.
        Persona Description:
        {
            "role": "Educator",
            "persona": "An experienced teacher who simplifies complex topics for teenagers.",
        }

        Task: Develop a new mobile app for tracking daily exercise.
        Persona Description:
        {
            "role": "Software Developer",
            "persona": "A creative developer with experience in mobile applications and user interface design.",
        }

        Task: Write a guide on how to cook Italian food for beginners.
        Persona Description:
        {
            "role": "Chef",
            "persona": "A professional chef specializing in Italian cuisine who enjoys teaching cooking techniques.",
        }
        """

    def generate_personas(self, task_description, num_agents):
        current_prompt = (
            self.base_prompt
            + f"\nNow, based on the schema above, generate a persona for the following task:\nTask: {task_description}\n"
        )

        agents = []
        while len(agents) < num_agents:
            # Send the prompt to the InferenceClient
            response = self.client.text_generation(
                current_prompt,
                max_new_tokens=400,
                grammar={"type": "json", "value": self.persona_grammar},
                stop_sequences=["<|eot_id|>"],
            )

            try:
                agents.append(json.loads(response))
            except json.decoder.JSONDecodeError:
                continue

            # Update the prompt with the newly generated persona for the next iteration
            if len(agents) == 0:
                current_prompt += "Already generated personas for this task:\n"
            current_prompt += f"\nPersona Description:\n{response}\n"

        return agents

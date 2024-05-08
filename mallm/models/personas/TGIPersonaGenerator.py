import json
import logging
from huggingface_hub import InferenceClient

from mallm.models.personas.PersonaGenerator import PersonaGenerator, Persona

logger = logging.getLogger("mallm")


class TGIPersonaGenerator(PersonaGenerator):
    def __init__(self, client: InferenceClient):
        self.client = client
        self.persona_grammar = {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["role", "description"],
        }
        self.base_prompt = """
When faced with a task, begin by identifying the participants who will contribute to solving the task. Provide role and description of the participants, describing their expertise or needs, formatted using the provided JSON schema.
Generate one participant at a time, complementing the existing participants to foster a rich discussion.

Example 1:
Task: Explain the basics of machine learning to high school students.
New Participant:
{"role": "Educator", "description": "An experienced teacher who simplifies complex topics for teenagers."}

Example 2:
Task: Develop a new mobile app for tracking daily exercise.
Already Generated Participants:
{"role": "Fitness Coach", "description": "A person that has high knowledge about sports and fitness."}
New Participant:
{"role": "Software Developer", "description": "A creative developer with experience in mobile applications and user interface design."}

Example 3:
Task: Write a guide on how to cook Italian food for beginners.
Already Generated Participants:
{"role": "Italian Native", "description": "An average home cook that lived in italy for 30 years."}
{"role": "Food Scientist", "description": "An educated scientist that knows which flavor combinations result in the best taste."}
New Participant:
{"role": "Chef", "description": "A professional chef specializing in Italian cuisine who enjoys teaching cooking techniques."}
        """

    def generate_personas(self, task_description, num_agents):
        current_prompt = (
            self.base_prompt
            + f"\nNow, based on the schema above, generate a participant to discuss the following task:\nTask: {task_description}\n"
        )

        logger.debug("Creating " + str(num_agents) + " agents...")
        agents = []
        while len(agents) < num_agents:
            # Send the prompt to the InferenceClient
            response = self.client.text_generation(
                current_prompt + "New Participant:\n",
                max_new_tokens=400,
                grammar={"type": "json", "value": self.persona_grammar},
                stop_sequences=["<|eot_id|>"],
                repetition_penalty=1.1,
            )

            try:
                agents.append(json.loads(response))
                logger.debug("Added one agent: " + str(agents[-1]))
            except json.decoder.JSONDecodeError as e:
                logger.error(
                    "Could not decode json: "
                    + str(e)
                    + "\nResponse string: "
                    + str(response)
                )
                continue

            # Update the prompt with the newly generated persona for the next iteration
            # We have to call this after the try catch to only add this when we successfully generate the first persona
            if len(agents) == 0:
                current_prompt += "Already Generated Participants:\n"
            current_prompt += f"{response}\n"

        return agents

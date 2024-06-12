import json
import logging

from mallm.models.Chat import Chat
from mallm.models.personas.PersonaGenerator import PersonaGenerator

logger = logging.getLogger("mallm")


class ExpertGenerator(PersonaGenerator):
    def __init__(self, llm: Chat):
        self.llm = llm
        self.base_prompt = {
            "role": "system",
            "content": """
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
        """,
        }

    def generate_personas(
        self, task_description: str, num_agents: int
    ) -> list[dict[str, str]]:
        current_prompt = [
            self.base_prompt,
            {
                "role": "user",
                "content": f"\nNow generate a participant to discuss the following task:\nTask: {task_description}\n",
            },
        ]

        logger.debug("Creating " + str(num_agents) + " personas...")
        agents: list[dict[str, str]] = []
        retry = 0
        while len(agents) < num_agents:
            # Send the prompt to the InferenceClient
            response = self.llm.invoke(
                [
                    *current_prompt,
                    {
                        "role": "user",
                        "content": "Please use the follow the examples to generate a useful persona for the task! Only answer with the JSON for the next persona!",
                    },
                ]
            )
            try:
                new_agent = json.loads(response)
                if new_agent["role"] == "" or new_agent["description"] == "":
                    continue
                agents.append(new_agent)
            except json.decoder.JSONDecodeError as e:
                retry = retry + 1
                logger.debug(
                    f"Could not decode json (will attempt retry no. {str(retry)}): "
                    + str(e)
                    + "\nResponse string: "
                    + str(response)
                )
                continue

            # Update the prompt with the newly generated persona for the next iteration
            current_prompt.append(
                {
                    "role": "system",
                    "content": f"Already Generated Participants:\n{response}",
                }
            )
        logger.debug("Found agents: \n" + str(agents))

        return agents
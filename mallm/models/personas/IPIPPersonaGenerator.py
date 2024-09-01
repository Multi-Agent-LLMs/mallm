import json
import logging

from mallm.models.Chat import Chat
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.utils.types import InputExample

logger = logging.getLogger("mallm")


class IPIPPersonaGenerator(PersonaGenerator):
    def __init__(self, llm: Chat):
        self.llm = llm
        self.base_prompt = {
            "role": "system",
            "content": """
When faced with a task, begin by identifying the participants who will contribute to solving the task. Provide role and fixed characteristics of the participant, formatted using the provided JSON schema.
Generate one participant at a time, complementing the existing participants to foster a rich discussion.

You must choose the following characteristics for the participant, in JSON format:
- "extraversion": "high" or "low"
- "egreeableness": "high" or "low"
- "conscientiousness": "high" or "low"
- "neuroticism": "high" or "low"
- "openness": "high" or "low"
- "experience": "Expert", "Neutral", "Non-Expert"
- "gender": "male", "female", "non-binary"

You absolutely must stick to the JSON format and the characteristics and options provided.

Example 1:
Task: Explain the basics of machine learning to high school students.
New Participant:
{"role": "Educator", "extraversion": "high", "egreeableness": "high", "conscientiousness": "high", "neuroticism": "low", "openness": "low", "experience": "Expert", "gender": "male"}

Example 2:
Task: Develop a new mobile app for tracking daily exercise.
Already Generated Participants:
{"role": "Fitness Coach", "extraversion": "high", "agreeableness": "low", "conscientiousness": "high", "neuroticism": "high", "openness": "low", "experience": "Neutral", "gender": "female"}
New Participant:
{"role": "Software Developer", "extraversion": "low", "agreeableness": "high", "conscientiousness": "high", "neuroticism": "low", "openness": "high", "experience": "Expert", "gender": "non-binary"}
        """,
        }

    def generate_persona(
        self, task_description: str, already_generated_personas: list[dict[str, str]], sample: InputExample
    ) -> dict[str, str]:
        current_prompt = [
            self.base_prompt,
            {
                "role": "user",
                "content": f"\nNow generate a participant to discuss the following task:\nTask: {task_description}\n",
            },
        ]

        retry = 0
        while retry < 5:
            # Send the prompt to the InferenceClient
            response = self.llm.invoke(
                [
                    *current_prompt,
                    {
                        "role": "user",
                        "content": "Only answer with the JSON for the next persona! Ensure your new participant is unique.",
                    },
                ]
            )

            try:
                new_agent = json.loads(response)

                # Check if all the required fields are present AND contain the valid options
                if "role" not in new_agent:
                    continue

                if "extraversion" not in new_agent or new_agent["extraversion"] not in {
                    "high",
                    "low",
                }:
                    continue

                if "agreeableness" not in new_agent or new_agent[
                    "agreeableness"
                ] not in {"high", "low"}:
                    continue

                if "conscientiousness" not in new_agent or new_agent[
                    "conscientiousness"
                ] not in {"high", "low"}:
                    continue

                if "neuroticism" not in new_agent or new_agent["neuroticism"] not in {
                    "high",
                    "low",
                }:
                    continue

                if "openness" not in new_agent or new_agent["openness"] not in {
                    "high",
                    "low",
                }:
                    continue

                if "experience" not in new_agent or new_agent["experience"] not in {
                    "Expert",
                    "Neutral",
                    "Non-Expert",
                }:
                    continue

                if "gender" not in new_agent or new_agent["gender"] not in {
                    "male",
                    "female",
                    "non-binary",
                }:
                    continue

                # Compose description for the agent using the attributes

                desc = ""
                # Extraversion
                desc += (
                    "You are extremely "
                    + ", extremely ".join(
                        [
                            "unfriendly",
                            "introverted",
                            "silent",
                            "timid",
                            "unassertive",
                            "inactive",
                            "unenergetic",
                            "unadventurous",
                            "gloomy",
                        ]
                        if new_agent["extraversion"] == "low"
                        else [
                            "friendly",
                            "extraverted",
                            "talkative",
                            "bold",
                            "assertive",
                            "active",
                            "energetic",
                            "adventurous",
                            "cheerful",
                        ]
                    )
                    + "."
                )

                # Agreeableness
                desc += (
                    " You are extremely "
                    + ", extremely ".join(
                        [
                            "distrustful",
                            "immoral",
                            "dishonest",
                            "unkind",
                            "stingy",
                            "unaltruistic",
                            "uncooperative",
                            "self-important",
                            "unsympathetic",
                            "selfish",
                            "disagreeable",
                        ]
                        if new_agent["agreeableness"] == "low"
                        else [
                            "trustful",
                            "moral",
                            "honest",
                            "kind",
                            "generous",
                            "altruistic",
                            "cooperative",
                            "humble",
                            "sympathetic",
                            "unselfish",
                            "agreeable",
                        ]
                    )
                    + "."
                )

                # Conscientiousness
                desc += (
                    " You are extremely "
                    + ", extremely ".join(
                        [
                            "unsure",
                            "messy",
                            "irresponsible",
                            "lazy",
                            "undisciplined",
                            "impractical",
                            "extravagant",
                            "disorganized",
                            "negligent",
                            "careless",
                        ]
                        if new_agent["conscientiousness"] == "low"
                        else [
                            "self-efficacious",
                            "orderly",
                            "responsible",
                            "hardworking",
                            "self-disciplined",
                            "practical",
                            "thrifty",
                            "organized",
                            "conscientious",
                            "thorough",
                        ]
                    )
                    + "."
                )

                # Neuroticism
                desc += (
                    " You are extremely "
                    + ", extremely ".join(
                        [
                            "relaxed",
                            "at ease",
                            "easygoing",
                            "calm",
                            "patient",
                            "happy",
                            "unselfconscious",
                            "level-headed",
                            "contented",
                            "emotionally stable",
                        ]
                        if new_agent["neuroticism"] == "low"
                        else [
                            "tense",
                            "nervous",
                            "anxious",
                            "angry",
                            "irritable",
                            "depressed",
                            "self-conscious",
                            "impulsive",
                            "discontented",
                            "emotionally unstable",
                        ]
                    )
                    + "."
                )

                # Openness to Experience
                desc += (
                    " You are extremely "
                    + ", extremely ".join(
                        [
                            "unimaginative",
                            "uncreative",
                            "artistically unappreciative",
                            "unaesthetic",
                            "unreflective",
                            "emotionally closed",
                            "uninquisitive",
                            "predictable",
                            "unintelligent",
                            "unanalytical",
                            "unsophisticated",
                            "socially conservative",
                        ]
                        if new_agent["openness"] == "low"
                        else [
                            "imaginative",
                            "creative",
                            "artistically appreciative",
                            "aesthetic",
                            "reflective",
                            "emotionally aware",
                            "curious",
                            "spontaneous",
                            "intelligent",
                            "analytical",
                            "sophisticated",
                            "socially progressive",
                        ]
                    )
                    + "."
                )

                desc += f" You are a {new_agent['experience']} in the field and identify as {new_agent['gender']}."

                agent : dict[str, str] = {"role": new_agent["role"], "description": desc}
                break
            except json.decoder.JSONDecodeError as e:
                logger.debug(
                    "Could not decode json (will attempt retry): "
                    + str(e)
                    + "\nResponse string: "
                    + str(response)
                )
                retry += 1
                continue

        return agent

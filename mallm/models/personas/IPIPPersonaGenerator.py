import json
import logging
from typing import Optional

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
- "agreeableness": "high" or "low"
- "conscientiousness": "high" or "low"
- "neuroticism": "high" or "low"
- "openness": "high" or "low"
- "experience": "Expert", "Neutral", "Non-Expert"
- "gender": "male", "female", "non-binary"

You absolutely must stick to the JSON format and the characteristics and options provided.

Example 1:
Task: Explain the basics of machine learning to high school students.
New Participant:
{"role": "Educator", "extraversion": "high", "agreeableness": "high", "conscientiousness": "high", "neuroticism": "low", "openness": "low", "experience": "Expert", "gender": "male"}

Example 2:
Task: Develop a new mobile app for tracking daily exercise.
Already Generated Participants:
{"role": "Fitness Coach", "extraversion": "high", "agreeableness": "low", "conscientiousness": "high", "neuroticism": "high", "openness": "low", "experience": "Neutral", "gender": "female"}
New Participant:
{"role": "Software Developer", "extraversion": "low", "agreeableness": "high", "conscientiousness": "high", "neuroticism": "low", "openness": "high", "experience": "Expert", "gender": "non-binary"}
        """,
        }

    def generate_personas(
        self, task_description: str, num_agents: int, sample: InputExample, extra_args: Optional[str] = None
    ) -> list[dict[str, str]]:
        current_prompt = [
            self.base_prompt,
            {
                "role": "user",
                "content": f"\nNow generate a participant to discuss the following task:\nTask: {task_description}\n",
            },
        ]

        logger.debug("Creating " + str(num_agents) + " agents...")
        agents: list[dict[str, str]] = []
        while len(agents) < num_agents:
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

            logger.debug("Persona Response: " + response)
            try:
                new_agent = json.loads(response)

                if extra_args:
                    if "e" in extra_args: # force low extraversion
                        new_agent["extraversion"] = "low"
                    if "a" in extra_args: # force low agreeableness
                        new_agent["agreeableness"] = "low"
                    if "c" in extra_args: # force low conscientiousness
                        new_agent["conscientiousness"] = "low"
                    if "n" in extra_args: # force low neuroticism
                        new_agent["neuroticism"] = "low"
                    if "o" in extra_args: # force low openness
                        new_agent["openness"] = "low"

                    if "E" in extra_args: # force high extraversion
                        new_agent["extraversion"] = "high"
                    if "A" in extra_args: # force high agreeableness
                        new_agent["agreeableness"] = "high"
                    if "C" in extra_args: # force high conscientiousness
                        new_agent["conscientiousness"] = "high"
                    if "N" in extra_args: # force high neuroticism
                        new_agent["neuroticism"] = "high"
                    if "O" in extra_args: # force high openness
                        new_agent["openness"] = "high"

                # Check if all the required fields are present AND contain the valid options
                if "role" not in new_agent:
                    logger.debug("Role not in new_agent")
                    continue

                if "extraversion" not in new_agent or new_agent["extraversion"] not in {
                    "high",
                    "low",
                }:
                    logger.debug("Extraversion not in new_agent")
                    continue

                if "agreeableness" not in new_agent or new_agent[
                    "agreeableness"
                ] not in {"high", "low"}:
                    logger.debug("Agreeableness not in new_agent")
                    continue

                if "conscientiousness" not in new_agent or new_agent[
                    "conscientiousness"
                ] not in {"high", "low"}:
                    logger.debug("Conscientiousness not in new_agent")
                    continue

                if "neuroticism" not in new_agent or new_agent["neuroticism"] not in {
                    "high",
                    "low",
                }:
                    logger.debug("Neuroticism not in new_agent")
                    continue

                if "openness" not in new_agent or new_agent["openness"] not in {
                    "high",
                    "low",
                }:
                    logger.debug("Openness not in new_agent")
                    continue

                if "experience" not in new_agent or new_agent["experience"] not in {
                    "Expert",
                    "Neutral",
                    "Non-Expert",
                }:
                    logger.debug("Experience not in new_agent")
                    continue

                if "gender" not in new_agent or new_agent["gender"] not in {
                    "male",
                    "female",
                    "non-binary",
                }:
                    logger.debug("Gender not in new_agent")
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

                my_agent = {
                    "role": new_agent["role"],
                    "description": desc,
                    "attributes": new_agent,
                }
                agents.append(my_agent)
            except json.decoder.JSONDecodeError as e:
                logger.debug(
                    "Could not decode json (will attempt retry): "
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

        return agents

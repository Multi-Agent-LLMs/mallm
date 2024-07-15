import logging

from mallm.models.Chat import Chat
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.utils.types import InputExample

logger = logging.getLogger("mallm")


class ParaphraseTypesGenerator(PersonaGenerator):
    def __init__(self, llm: Chat):
        self.llm = llm
        self.paraphrase_types = (
            {  # Paraphrase types from ETPC: https://aclanthology.org/L18-1221.pdf
                "name1": "desc1",
                "name2": "desc2",
            }
        )

    def generate_personas(
        self, task_description: str, num_agents: int, sample: InputExample
    ) -> list[dict[str, str]]:
        if not sample.context:
            logger.error(
                "Failed to generate personas because there were no paraphrase types provided in the context."
            )
            raise Exception("Failed to generate personas.")

        context = sample.context[0].replace("Paraphrase Types: ", "")
        paraphrase_types_list = context.split(",")
        print(paraphrase_types_list)
        logger.debug("Creating " + str(len(paraphrase_types_list)) + " personas...")
        agents = [
            {"role": p, "description": self.paraphrase_types[p]}
            for p in paraphrase_types_list
        ]
        logger.debug("Found agents: \n" + str(agents))

        return agents

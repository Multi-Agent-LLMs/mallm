import dbm
import json
import logging
import os
import time
import uuid
from datetime import timedelta

import fire
import transformers

from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.decision_making.DecisionProtocol import DecisionProtocol
from mallm.decision_making.MajorityConsensus import MajorityConsensus
from mallm.decision_making.Voting import Voting
from mallm.discourse_policy.DiscourceDebate import DiscourseDebate
from mallm.discourse_policy.DiscourceMemory import DiscourseMemory
from mallm.discourse_policy.DiscourceRelay import DiscourseRelay
from mallm.discourse_policy.DiscourceReport import DiscourseReport
from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.prompts.coordinator_prompts import generate_chat_prompt_extract_result
from mallm.utils.types.Agreement import Agreement

transformers.logging.set_verbosity_error()
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

logger = logging.getLogger("mallm")


class Coordinator:

    def __init__(
        self,
        model,
        client,
        agent_generator: PersonaGenerator = None,
        use_moderator=False,
        memory_bucket_dir="./mallm/utils/memory_bucket/",
    ):
        self.personas = None
        self.id = str(uuid.uuid4())
        self.short_id = self.id[:4]
        self.panelists = []
        self.agents = []
        self.use_moderator = use_moderator
        self.moderator = None
        self.memory_bucket_dir = memory_bucket_dir
        self.memory_bucket = self.memory_bucket_dir + "global_" + self.id
        self.decision_making: DecisionProtocol = None
        self.llm = model
        self.client = client
        self.agent_generator = agent_generator

    def init_agents(self, task_instruction, input_str, use_moderator):
        """
        Instantiates the agents by
        1) identify helpful personas
        2) create agents with the personas
        Gives true if the automatic assignment was successfull.
        Returns bool
        """
        self.panelists = []
        self.agents = []

        personas = self.agent_generator.generate_personas(
            f"{task_instruction} {input_str}", 3
        )

        if use_moderator:
            self.moderator = Moderator(self.llm, self.client, self)
        for persona in personas:
            self.panelists.append(
                Panelist(
                    self.llm, self.client, self, persona["role"], persona["description"]
                )
            )

        if use_moderator:
            self.agents = [self.moderator] + self.panelists
        else:
            self.agents = self.panelists
        return True

    def get_agents(self):
        agent_dicts = []
        for a in self.agents:
            agent_dicts.append(
                {
                    "agentId": a.id,
                    "model": "placeholder",  # TODO: automatically detect model name
                    "persona": a.persona,
                    "personaDescription": a.persona_description,
                }
            )
        return agent_dicts

    def update_global_memory(
        self,
        unique_id,
        turn,
        agent_id,
        persona,
        contribution,
        text,
        agreement,
        extracted_draft,
        memory_ids,
        prompt_args,
    ):
        """
        Updates the dbm memory with another discussion entry.
        Returns string
        """

        data_dict = {
            "messageId": unique_id,
            "turn": turn,
            "agentId": agent_id,
            "persona": str(persona).replace('"', "'"),
            "additionalArgs": prompt_args,
            "contribution": contribution,
            "memoryIds": memory_ids,
            "text": str(text).replace('"', "'"),
            "agreement": agreement,
            "extractedDraft": str(extracted_draft).replace('"', "'"),
        }

        with dbm.open(self.memory_bucket, "c") as db:
            db[str(unique_id)] = json.dumps(data_dict)
            logger.debug(str(db[str(unique_id)]))
        self.save_global_memory_to_json()

    def get_global_memory(self):
        """
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        """
        memory = []
        with dbm.open(self.memory_bucket, "r") as db:
            for key in db.keys():
                memory.append(json.loads(db[key].decode()))
        return memory

    def save_global_memory_to_json(self):
        """
        Converts the memory bucket dbm data to json format
        """
        try:
            with open(self.memory_bucket + ".json", "w") as f:
                json.dump(self.get_global_memory(), f)
        except Exception as e:
            logger.error(f"Failed to save agent memory to {self.memory_bucket}: {e}")
            logger.error(self.get_global_memory())

    def update_memories(self, memories, agents_to_update):
        """
        Updates the memories of all declared agents.
        """
        for c in memories:
            for a in agents_to_update:
                a.update_memory(
                    c["messageId"],
                    c["turn"],
                    c["agentId"],
                    c["persona"],
                    c["contribution"],
                    c["text"],
                    c["agreement"],
                    c["extractedDraft"],
                    c["memoryIds"],
                    c["additionalArgs"],
                )
        return []

    def discuss(
        self,
        task_instruction,
        input_str,
        context,
        use_moderator,
        feedback_sentences=[3, 4],
        paradigm="memory",
        max_turns=None,
        context_length=1,
        include_current_turn_in_memory=False,
        extract_all_drafts=False,
        debate_rounds=1,
    ) -> tuple[str, list, list, int, list[Agreement], float]:
        """
        The routine responsible for the discussion between agents to solve a task.

        The routine is organized as follows:
        1) Create agents with personas
        2) Discuss the problem based on the given paradigm (iteratively check for agreement between agents)
        3) After max turns or agreement reached: return the final result to the task sample

        Returns the final response agreed on, the global memory, agent specific memory, turns needed, last agreements of agents
        """
        if context:
            if isinstance(context, list):
                for c in context:
                    task_instruction += "\n" + c
            elif isinstance(context, str):
                task_instruction += "\n" + context

        if not self.init_agents(
            task_instruction, input_str, use_moderator=use_moderator
        ):
            logger.error(f"""Failed to intialize agents (coordinator: {self.id}).""")
            return (
                None,
                None,
                None,
                None,
                None,
                None,
            )  # if the LLM failed to initialize the agents, do not discuss

        personas = [a.persona for a in self.agents]
        if len(personas) <= 2:
            logger.error(
                "Only two or less personas were generated. No discussion is executed."
            )
            return (
                None,
                None,
                None,
                None,
                None,
                None,
            )  # if the LLM failed to initialize the agents, do not discuss

        self.decision_making: DecisionProtocol = Voting(self.panelists)

        logger.info(
            f"""
Starting discussion with coordinator {self.id}...
-------------
Instruction: {task_instruction}
Input: {input_str}
Feedback sentences: {str(feedback_sentences)}
Maximum turns: {max_turns}
Agents: {str(personas)}
Decision-making: {self.decision_making.__class__.__name__}
-------------"""
        )

        startTime = time.perf_counter()
        protocols = {
            "memory": DiscourseMemory,
            "report": DiscourseReport,
            "relay": DiscourseRelay,
            "debate": DiscourseDebate,
        }
        if paradigm not in protocols:
            logger.error(f"No valid discourse policy for paradigm {paradigm}")
            exit(-1)
        policy: DiscoursePolicy = protocols[paradigm]()

        current_draft, turn, agreements = policy.discuss(
            self,
            task_instruction,
            input_str,
            use_moderator,
            feedback_sentences,
            max_turns,
            context_length,
            include_current_turn_in_memory,
            extract_all_drafts,
        )

        discussion_time = timedelta(
            seconds=time.perf_counter() - startTime
        ).total_seconds()

        global_mem = self.get_global_memory()
        agent_mems = []
        for a in self.agents:
            agent_mems.append(a.get_memory()[0])
        if turn >= max_turns:  # if no agreement was reached
            current_draft = None
        else:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(input_str, current_draft),
                client=self.client,
            )

        return current_draft, global_mem, agent_mems, turn, agreements, discussion_time


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)

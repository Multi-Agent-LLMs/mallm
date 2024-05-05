import time
from datetime import timedelta

import transformers
from langchain.chains import LLMChain

from mallm.agents.moderator import *
from mallm.agents.panelist import *
from mallm.decision_making.consensus import *
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.prompts import coordinator_prompts

transformers.logging.set_verbosity_error()
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

logger = logging.getLogger("mallm")


class Coordinator:

    def __init__(
        self,
        model,
        client,
        agent_generator: PersonaGenerator,
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
        self.decision_making = None
        self.llm = model
        self.client = client
        self.agent_generator = agent_generator

        self.chain_baseline = LLMChain(
            llm=self.llm, prompt=coordinator_prompts.baseline
        )

    def initAgents(self, task_instruction, input, use_moderator):
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
            f"{task_instruction} {input}", 3
        )

        if use_moderator:
            self.moderator = Moderator(self.llm, self.client, self)
        for persona in personas:
            self.panelists.append(
                Panelist(
                    self.llm, self.client, persona["role"], persona["persona"], self
                )
            )

        if use_moderator:
            self.agents = [self.moderator] + self.panelists
        else:
            self.agents = self.panelists
        return True

    def getAgents(self):
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

    def updateGlobalMemory(
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

    def discuss_memory(
        self,
        task_instruction,
        input,
        use_moderator,
        feedback_sentences=[3, 4],
        max_turns=None,
        context_length=1,
        include_current_turn_in_memory=False,
        extract_all_drafts=False,
    ):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

        logger.debug(
            """Paradigm: Memory
                    ┌───┐
                    │A 1│
                    ├───┘
                    │   ▲
                    │   │
                    ▼   │
        ┌───┬──────►┌───┤◄──────┬───┐
        │A 3│       │MEM│       │A 2│
        └───┘◄──────┴───┴──────►└───┘
        """
        )
        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            logger.info(
                "Discussion " + self.id + " ongoing. Current turn: " + str(turn)
            )

            if use_moderator:
                memory_string, memory_ids, current_draft = (
                    self.moderator.get_memory_string(
                        context_length=context_length,
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory,
                    )
                )

                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.moderator.persona,
                    "personaDescription": self.moderator.persona_description,
                    "agentMemory": memory_string,
                }
                res, memory, agreements = self.moderator.draft(
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    agreements,
                    is_moderator=True,
                )
                memories.append(memory)
                memories = self.update_memories(memories, self.agents)
                unique_id = unique_id + 1

            for p in self.panelists:
                memory_string, memory_ids, current_draft = p.get_memory_string(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": p.persona,
                    "personaDescription": p.persona_description,
                    "sentsMin": feedback_sentences[0],
                    "sentsMax": feedback_sentences[1],
                    "agentMemory": memory_string,
                }

                memories, agreements = p.participate(
                    use_moderator,
                    memories,
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    self.agents,
                    agreements,
                )
                unique_id = unique_id + 1

            decision = self.decision_making.decide(agreements, turn)

        return current_draft, turn, agreements

    def discuss_report(
        self,
        task_instruction,
        input,
        use_moderator,
        feedback_sentences=[3, 4],
        max_turns=None,
        context_length=1,
        include_current_turn_in_memory=False,
        extract_all_drafts=False,
    ):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

        logger.debug(
            """Paradigm: Report
                    ┌───┐
                    │A 1│
            ┌──────►└┼─┼┘◄──────┐
            │        │ │        │
            │        │ │        │
            │        │ │        │
        ┌───┼◄───────┘ └───────►├───┐
        │A 3│                   │A 2│
        └───┘                   └───┘
        """
        )

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            logger.info("Ongoing. Current turn: " + str(turn))

            # ---- Agent A1
            if use_moderator:
                memory_string, memory_ids, current_draft = (
                    self.moderator.get_memory_string(
                        context_length=context_length,
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory,
                    )
                )

                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.moderator.persona,
                    "personaDescription": self.moderator.persona_description,
                    "agentMemory": memory_string,
                }
                res, memory, agreements = self.moderator.draft(
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    agreements,
                    is_moderator=True,
                )
                memories.append(memory)
                memories = self.update_memories(memories, self.agents)
                unique_id = unique_id + 1
            else:
                memory_string, memory_ids, current_draft = self.panelists[
                    0
                ].get_memory_string(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.panelists[0].persona,
                    "personaDescription": self.panelists[0].persona_description,
                    "agentMemory": memory_string,
                }
                res, memory, agreements = self.panelists[0].draft(
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    agreements,
                )
                memories.append(memory)
                memories = self.update_memories(memories, self.agents)
                unique_id = unique_id + 1

            # ---- Agents A2, A3, A4, ...
            for p in self.agents[1:]:
                memory_string, memory_ids, current_draft = p.get_memory_string(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": p.persona,
                    "personaDescription": p.persona_description,
                    "sentsMin": feedback_sentences[0],
                    "sentsMax": feedback_sentences[1],
                    "agentMemory": memory_string,
                }

                memories, agreements = p.participate(
                    True,
                    memories,
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    [self.agents[0], p],
                    agreements,
                )
                unique_id = unique_id + 1

            decision = self.decision_making.decide(agreements, turn)
        return current_draft, turn, agreements

    def discuss_relay(
        self,
        task_instruction,
        input,
        use_moderator,
        feedback_sentences=[3, 4],
        max_turns=None,
        context_length=1,
        include_current_turn_in_memory=False,
        extract_all_drafts=False,
    ):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

        logger.debug(
            """Paradigm: Relay
                    ┌───┐
          ┌────────►│A 1│─────────┐
          │         └───┘         │
          │                       │
          │                       │
          │                       ▼
        ┌─┴─┐                   ┌───┐
        │A 3│◄──────────────────┤A 2│
        └───┘                   └───┘
        """
        )

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            logger.info("Ongoing. Current turn: " + str(turn))

            for i, a in enumerate(self.agents):
                memory_string, memory_ids, current_draft = a.get_memory_string(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                next_a = i + 1
                if i == len(self.agents) - 1:
                    next_a = 0  # start again with agent 0 (loop)
                if a == self.moderator:
                    template_filling = {
                        "task_instruction": task_instruction,
                        "input": input,
                        "current_draft": current_draft,
                        "persona": self.moderator.persona,
                        "persona_description": self.moderator.persona_description,
                        "agent_memory": memory_string,
                    }
                    res, memory, agreements = self.moderator.draft(
                        unique_id,
                        turn,
                        memory_ids,
                        template_filling,
                        extract_all_drafts,
                        agreements,
                        is_moderator=True,
                    )
                    memories.append(memory)
                    memories = self.update_memories(memories, [a, self.agents[next_a]])
                else:
                    template_filling = {
                        "taskInstruction": task_instruction,
                        "input": input,
                        "currentDraft": current_draft,
                        "persona": a.persona,
                        "personaDescription": a.persona_description,
                        "sentsMin": feedback_sentences[0],
                        "sentsMax": feedback_sentences[1],
                        "agentMemory": memory_string,
                    }
                    memories, agreements = a.participate(
                        use_moderator,
                        memories,
                        unique_id,
                        turn,
                        memory_ids,
                        template_filling,
                        extract_all_drafts,
                        [a, self.agents[next_a]],
                        agreements,
                    )
                unique_id = unique_id + 1

            decision = self.decision_making.decide(agreements, turn)

        return current_draft, turn, agreements

    def discuss_debate(
        self,
        task_instruction,
        input,
        use_moderator,
        feedback_sentences=[3, 4],
        max_turns=None,
        context_length=1,
        include_current_turn_in_memory=False,
        extract_all_drafts=False,
        debate_rounds=1,
    ):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

        logger.debug(
            f"""Paradigm: Debate (rounds: {debate_rounds})
                    ┌───┐
          ┌────────►│A 1│◄────────┐
          │         └───┘         │
          │                       │
          │                       │
          │                       │
        ┌─┴─┬──────────────────►┌─┴─┐
        │A 3│                   │A 2│
        └───┘◄──────────────────┴───┘
        """
        )

        logger.info("Debate rounds between agents A2, ..., An: " + str(debate_rounds))

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            log = "Ongoing. Current turn: " + str(turn)
            logger.info("Ongoing. Current turn: " + str(turn))

            # ---- Agent A1
            if use_moderator:
                memory_string, memory_ids, current_draft = (
                    self.moderator.get_memory_string(
                        context_length=context_length,
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory,
                    )
                )

                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.moderator.persona,
                    "personaDescription": self.moderator.persona_description,
                    "agentMemory": memory_string,
                }
                res, memory, agreements = self.moderator.draft(
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    agreements,
                    is_moderator=True,
                )
                memories.append(memory)
                memories = self.update_memories(memories, self.agents)
                unique_id = unique_id + 1
            else:
                memory_string, memory_ids, current_draft = self.panelists[
                    0
                ].get_memory_string(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory,
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.panelists[0].persona,
                    "personaDescription": self.panelists[0].persona_description,
                    "agentMemory": memory_string,
                }
                res, memory, agreements = self.panelists[0].draft(
                    unique_id,
                    turn,
                    memory_ids,
                    template_filling,
                    extract_all_drafts,
                    agreements,
                    is_moderator=True,
                )
                memories.append(memory)
                memories = self.update_memories(memories, self.agents)
                unique_id = unique_id + 1

            for r in range(debate_rounds):  # ---- Agents A2, A3, ...
                logger.debug("Debate round: " + str(r))
                debate_agreements = []
                for i, a in enumerate(self.agents[1:]):  # similar to relay paradigm
                    memory_string, memory_ids, current_draft = a.get_memory_string(
                        context_length=context_length,
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory,
                    )
                    next_a = i + 2
                    if i == len(self.agents[1:]) - 1:
                        next_a = 1  # start again with agent 1 (loop)

                    template_filling = {
                        "taskInstruction": task_instruction,
                        "input": input,
                        "currentDraft": current_draft,
                        "persona": a.persona,
                        "personaDescription": a.persona_description,
                        "sentsMin": feedback_sentences[0],
                        "sentsMax": feedback_sentences[1],
                        "agentMemory": memory_string,
                    }
                    if r == debate_rounds - 1:  # last debate round
                        agents_to_update = [self.agents[0], a, self.agents[next_a]]
                    else:
                        agents_to_update = [a, self.agents[next_a]]
                    memories, debate_agreements = a.participate(
                        use_moderator,
                        memories,
                        unique_id,
                        turn,
                        memory_ids,
                        template_filling,
                        extract_all_drafts,
                        agents_to_update,
                        debate_agreements,
                    )
                    if len(debate_agreements) > len(self.agents) - 1:
                        debate_agreements = debate_agreements[1 - len(self.agents) :]
                    unique_id = unique_id + 1

            agreements = agreements + debate_agreements
            if len(agreements) > len(self.panelists):
                agreements = agreements[-len(self.panelists) :]
            decision = self.decision_making.decide(agreements, turn)

        return current_draft, turn, agreements

    def discuss(
        self,
        task_instruction,
        input,
        context,
        use_moderator,
        feedback_sentences=[3, 4],
        paradigm="memory",
        max_turns=None,
        context_length=1,
        include_current_turn_in_memory=False,
        extract_all_drafts=False,
        debate_rounds=1,
    ):
        """
        The routine responsible for the discussion between agents to solve a task.

        The routine is organized as follows:
        1) Create agents with personas
        2) Discuss the problem based on the given paradigm (iteratively check for agreement between agents)
        3) After max turns or agreement reached: return the final result to the task sample

        Returns the final response agreed on, the global memory, agent specific memory, turns needed, last agreements of agents
        """
        if context:
            task_instruction += "\n" + "Context: " + context

        if not self.initAgents(task_instruction, input, use_moderator=use_moderator):
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

        self.decision_making = MajorityConsensus(self.panelists)

        logger.info(
            f"""
Starting discussion with coordinator {self.id}...
-------------
Instruction: {task_instruction}
Input: {input}
Feedback sentences: {str(feedback_sentences)}
Maximum turns: {max_turns}
Agents: {str(personas)}
Decision-making: {self.decision_making.__class__.__name__}
-------------"""
        )

        startTime = time.perf_counter()
        if paradigm == "memory":
            current_draft, turn, agreements = self.discuss_memory(
                task_instruction,
                input,
                use_moderator,
                feedback_sentences,
                max_turns,
                context_length,
                include_current_turn_in_memory,
                extract_all_drafts,
            )
        elif paradigm == "report":
            current_draft, turn, agreements = self.discuss_report(
                task_instruction,
                input,
                use_moderator,
                feedback_sentences,
                max_turns,
                context_length,
                include_current_turn_in_memory,
                extract_all_drafts,
            )
        elif paradigm == "relay":
            current_draft, turn, agreements = self.discuss_relay(
                task_instruction,
                input,
                use_moderator,
                feedback_sentences,
                max_turns,
                context_length,
                include_current_turn_in_memory,
                extract_all_drafts,
            )
        elif paradigm == "debate":
            current_draft, turn, agreements = self.discuss_debate(
                task_instruction,
                input,
                use_moderator,
                feedback_sentences,
                max_turns,
                context_length,
                include_current_turn_in_memory,
                extract_all_drafts,
                debate_rounds,
            )
        discussionTime = timedelta(
            seconds=time.perf_counter() - startTime
        ).total_seconds()

        globalMem = self.get_global_memory()
        agentMems = []
        for a in self.agents:
            agentMems.append(a.get_memory()[0])
        if turn >= max_turns:  # if no agreement was reached
            current_draft = None
        else:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(current_draft), client=self.client
            )

        return current_draft, globalMem, agentMems, turn, agreements, discussionTime


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)

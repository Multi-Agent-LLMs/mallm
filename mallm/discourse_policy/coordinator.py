import glob
import re
import time
from datetime import timedelta

import transformers
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from torch import cuda, bfloat16

from mallm.agents.moderator import *
from mallm.agents.panelist import *
from mallm.decision_making.consensus import *
from mallm.prompts import coordinator_prompts

transformers.logging.set_verbosity_error()
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


class Coordinator():
    def __init__(self, use_moderator=True, verbose=False):
        self.personas = None
        self.panelists = []
        self.agents = []
        self.use_moderator = use_moderator
        self.moderator = None
        self.memory_bucket = memory_bucket_dir + "global"
        self.decision_making = None
        self.llm_tokenizer = None
        self.llm = self.create_llm()
        self.init_chains()
        self.verbose = verbose

    def init_chains(self):
        if "llama" in self.llm_tokenizer.__class__.__name__.lower():  # use <<SYS>> and [INST] tokens for llama models
            partial_variables = {"sys_s": "<<SYS>>", "sys_e": "<</SYS>>", "inst_s": "[INST]", "inst_e": "[/INST]"}
        else:
            partial_variables = {"sys_s": "", "sys_e": "", "inst_s": "", "inst_e": ""}

        self.chain_identify_personas = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
            template=coordinator_prompts.identify_personas(),
            partial_variables=partial_variables))
        self.chain_extract_result = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
            template=coordinator_prompts.extract_result(),
            partial_variables=partial_variables))
        self.chain_baseline = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
            template=coordinator_prompts.baseline(),
            partial_variables=partial_variables))

    def initAgents(self, task_instruction, input, use_moderator, persona_type="expert"):
        '''
        Instantiates the agents by
        1) identify helpful personas
        2) create agents with the personas
        Gives true if the automatic assignment was successfull.
        Returns bool
        '''
        self.personas = None
        self.panelists = []
        self.agents = []
        template_filling = {
            "taskInstruction": task_instruction,
            "input": input
        }

        res = self.chain_identify_personas.invoke(template_filling)["text"]
        if self.verbose:
            print(res)

        # TODO: Use grammar to force LLM output in the correct JSON format. Example with llama.ccp: https://til.simonwillison.net/llms/llama-cpp-python-grammars

        # repair dictionary in string if the LLM did mess up the formatting
        if "{" in res and "}" not in res:
            print(
                "Looks like the LLM did not provide a valid dictionary (maybe the last brace is missing?). Trying to repair the dictionary...")
            res = res + "}"

        # self.updateGlobalMemory(0, 0, None, None, "persona_identification", res, None, [], template_filling)

        personas_string = re.search(r"\{.*?\}", res, re.DOTALL)
        if not personas_string:
            print(f"LLM failed to provide personas in the correct format - Skipping this sample...")
            # personas_string = '''{
            #    "Poet": "A person who studies and creates poetry. The poet is familiar with the rules and formats of poetry and can provide guidance on how to write a poem.",
            #    "Computer Scientist": "A scholar who specializes in the academic study of computer science. The computer scientist is familiar with the concept of a quantum computer and can provide guidance on how to explain it.",
            #    "Ten year old child": "A child with a limited English vocabulary and little knowledge about complicated concepts, such as a quantum computer."
            #    }'''
            # self.personas = ast.literal_eval(personas_string)
            return False
        else:
            personas_string = personas_string.group()
            for i in [0, 1]:
                try:
                    self.personas = ast.literal_eval(personas_string)
                except Exception as e:
                    if i == 0:
                        print(
                            "Looks like the LLM did not get the formatting quite right. Trying to repair the dictionary string...")
                        personas_string = personas_string.replace("\'\n", "\',\n")
                        personas_string = personas_string.replace('\"\n', '\",\n')
                        personas_string = personas_string.replace('\";', '\",')
                        print("Repaired string: \n" + str(personas_string))
                        continue
                    elif i == 1:
                        print(f"Failed to parse the string to identify personas: {e} - Skipping this sample...")
                        # personas_string = '''{
                        # "Poet": "A person who studies and creates poetry. The poet is familiar with the rules and formats of poetry and can provide guidance on how to write a poem.",
                        # "Computer Scientist": "A scholar who specializes in the academic study of computer science. The computer scientist is familiar with the concept of a quantum computer and can provide guidance on how to explain it.",
                        # "Ten year old child": "A child with a limited English vocabulary and little knowledge about complicated concepts, such as a quantum computer."
                        # }'''
                        # self.personas = ast.literal_eval(personas_string)
                        return False

        self.panelists = []
        if use_moderator:
            self.moderator = Moderator(self.llm, self.llm_tokenizer, self)
        for i, p in enumerate(self.personas):
            self.panelists.append(Panelist(self.llm, self.llm_tokenizer, p, self.personas[p], self))

        if use_moderator:
            self.personas[self.moderator.persona] = self.moderator.persona_description
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
                    "model": ckpt_dir.split("/")[-1],
                    "persona": a.persona,
                    "personaDescription": a.persona_description
                }
            )
        return agent_dicts
    
    def create_llm(self):
        '''
        Initializes the LLM that the agents are using to generate their outputs. 
        The LLM is set in evaluation mode. Thus, it immediately forgets everything that happened. 
        It allows for an all-fresh reprompting at each iteration of the discussion.
        Any model within the huggingface format can be loaded.
        Returns HuggingFacePipeline
        '''
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        print(f"Running on device: {device}")
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        model_config = transformers.AutoConfig.from_pretrained(
            ckpt_dir
        )
        if device == "cpu":  # not recommended but useful for developing with no GPU available
            model = transformers.AutoModelForCausalLM.from_pretrained(
                ckpt_dir,
                trust_remote_code=True,
                config=model_config,
                offload_folder="offload",
                device_map='auto'
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                ckpt_dir,
                trust_remote_code=True,
                config=model_config,
                quantization_config=bnb_config,
                device_map='auto'
            )
        model.eval()
        print(f"Model {ckpt_dir} loaded on {device}")

        self.llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
            ckpt_dir
        )
        # self.llm_tokenizer.pad_token_id = model.config.eos_token_id
        print("Using this tokenizer: " + str(self.llm_tokenizer.__class__.__name__))

        pipeline = transformers.pipeline(
            model=model,
            tokenizer=self.llm_tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            pad_token_id=self.llm_tokenizer.eos_token_id,
            # model parameters
            do_sample=True,
            temperature=0.9,
            max_new_tokens=512,  # max number of tokens to generate in the output
            min_new_tokens=2,  # always answer something (no empty responses)
            repetition_penalty=1.1,  # without this output begins repeating
        )

        return HuggingFacePipeline(pipeline=pipeline)

    def updateGlobalMemory(self, unique_id, turn, agent_id, persona, contribution, text, agreement, extracted_draft,
                           memory_ids, prompt_args):
        '''
        Updates the dbm memory with another discussion entry.
        Returns string
        '''
        with dbm.open(self.memory_bucket, 'c') as db:
            db[
                str(unique_id)] = f'''{{"messageId": {unique_id}, "turn": {turn}, "agentId": "{agent_id}", "persona": "{str(persona).replace('"', "'")}", "additionalArgs":{prompt_args}, "contribution": "{contribution}", "memoryIds": {memory_ids}, "text": "{str(text).replace('"', "'")}", "agreement": {agreement}, "extractedDraft": "{str(extracted_draft).replace('"', "'")}"}}'''
            if self.verbose:
                print(str(db[str(unique_id)]))  # logging
        self.saveGlobalMemoryToJson()

    def getGlobalMemory(self):
        '''
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        '''
        memory = []
        with dbm.open(self.memory_bucket, 'r') as db:
            for key in db.keys():
                memory.append(ast.literal_eval(db[key].decode().replace("\n", "\\n").replace("\t", "\\t")))
        return memory

    def saveGlobalMemoryToJson(self):
        '''
        Converts the memory bucket dbm data to json format
        '''
        try:
            with open(self.memory_bucket + ".json", "w") as f:
                json.dump(self.getGlobalMemory(), f)
        except Exception as e:
            print(f"Failed to save agent memory to {self.memory_bucket}: {e}")
            print(self.getGlobalMemory())

    def cleanMemoryBucket(self):
        '''
        Deletes all stored global memory
        '''
        filelist = glob.glob(os.path.join(memory_bucket_dir, "*.bak"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(memory_bucket_dir, "*.dat"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(memory_bucket_dir, "*.dir"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(memory_bucket_dir, "*.json"))
        for f in filelist:
            os.remove(f)
        print("Cleaned the memory bucket.")

    def updateMemories(self, memories, agents_to_update):
        '''
        Updates the memories of all declared agents.
        '''
        for c in memories:
            for a in agents_to_update:
                a.updateMemory(c["messageId"], c["turn"], c["agentId"], c["persona"], c["contribution"], c["text"],
                               c["agreement"], c["extractedDraft"], c["memoryIds"], c["additionalArgs"])
        return []

    def discuss_memory(self, task_instruction, input, use_moderator, feedback_sentences=[3, 4],
                max_turns=None, context_length=1, include_current_turn_in_memory=False, extract_all_drafts=False):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

        print('''Paradigm: Memory
                    ┌───┐
                    │A 1│
                    ├───┘
                    │   ▲
                    │   │
                    ▼   │
        ┌───┬──────►┌───┤◄──────┬───┐
        │A 3│       │MEM│       │A 2│
        └───┘◄──────┴───┴──────►└───┘
        ''')
        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            log = "Ongoing. Current turn: " + str(turn)
            if not self.verbose:
                log = "\r" + log + "        "
            print(log, end='')

            if use_moderator:
                memory_string, memory_ids, current_draft = self.moderator.getMemoryString(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory
                )

                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.moderator.persona,
                    "personaDescription": self.moderator.persona_description,
                    "agentMemory": memory_string
                }
                res, memory, agreements = self.moderator.draft(unique_id, turn, memory_ids, template_filling,
                                                                extract_all_drafts, agreements, is_moderator=True)
                memories.append(memory)
                memories = self.updateMemories(memories, self.agents)
                unique_id = unique_id + 1

            for p in self.panelists:
                memory_string, memory_ids, current_draft = p.getMemoryString(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": p.persona,
                    "personaDescription": p.persona_description,
                    "sentsMin": feedback_sentences[0],
                    "sentsMax": feedback_sentences[1],
                    "agentMemory": memory_string
                }

                memories, agreements = p.participate(use_moderator, memories, unique_id, turn, memory_ids,
                                                        template_filling, extract_all_drafts, self.agents, agreements)
                unique_id = unique_id + 1

            decision = self.decision_making.decide(agreements, turn)

        return current_draft, turn, agreements

    def discuss_report(self, task_instruction, input, use_moderator, feedback_sentences=[3, 4],
                max_turns=None, context_length=1, include_current_turn_in_memory=False, extract_all_drafts=False):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

        print('''Paradigm: Report
                    ┌───┐
                    │A 1│
            ┌──────►└┼─┼┘◄──────┐
            │        │ │        │
            │        │ │        │
            │        │ │        │
        ┌───┼◄───────┘ └───────►├───┐
        │A 3│                   │A 2│
        └───┘                   └───┘
        ''')

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            log = "Ongoing. Current turn: " + str(turn)
            if not self.verbose:
                log = "\r" + log + "        "
            print(log, end='')

            # ---- Agent A1
            if use_moderator:
                memory_string, memory_ids, current_draft = self.moderator.getMemoryString(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory
                )

                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.moderator.persona,
                    "personaDescription": self.moderator.persona_description,
                    "agentMemory": memory_string
                }
                res, memory, agreements = self.moderator.draft(unique_id, turn, memory_ids, template_filling,
                                                                extract_all_drafts, agreements, is_moderator=True)
                memories.append(memory)
                memories = self.updateMemories(memories, self.agents)
                unique_id = unique_id + 1
            else:
                memory_string, memory_ids, current_draft = self.panelists[0].getMemoryString(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.panelists[0].persona,
                    "personaDescription": self.panelists[0].persona_description,
                    "agentMemory": memory_string
                }
                res, memory, agreements = self.panelists[0].draft(unique_id, turn, memory_ids, template_filling,
                                                                    extract_all_drafts, agreements)
                memories.append(memory)
                memories = self.updateMemories(memories, self.agents)
                unique_id = unique_id + 1

            # ---- Agents A2, A3, A4, ...
            for p in self.agents[1:]:
                memory_string, memory_ids, current_draft = p.getMemoryString(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": p.persona,
                    "personaDescription": p.persona_description,
                    "sentsMin": feedback_sentences[0],
                    "sentsMax": feedback_sentences[1],
                    "agentMemory": memory_string
                }

                memories, agreements = p.participate(True, memories, unique_id, turn, memory_ids, template_filling,
                                                        extract_all_drafts, [self.agents[0], p], agreements)
                unique_id = unique_id + 1

            decision = self.decision_making.decide(agreements, turn)
        return current_draft, turn, agreements

    def discuss_relay(self, task_instruction, input, use_moderator, feedback_sentences=[3, 4],
                max_turns=None, context_length=1, include_current_turn_in_memory=False, extract_all_drafts=False):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

        print('''Paradigm: Relay
                    ┌───┐
          ┌────────►│A 1│─────────┐
          │         └───┘         │
          │                       │
          │                       │
          │                       ▼
        ┌─┴─┐                   ┌───┐
        │A 3│◄──────────────────┤A 2│
        └───┘                   └───┘
        ''')

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            log = "Ongoing. Current turn: " + str(turn)
            if not self.verbose:
                log = "\r" + log + "        "
            print(log, end='')

            for i, a in enumerate(self.agents):
                memory_string, memory_ids, current_draft = a.getMemoryString(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory
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
                        "agent_memory": memory_string
                    }
                    res, memory, agreements = self.moderator.draft(unique_id, turn, memory_ids, template_filling,
                                                                    extract_all_drafts, agreements,
                                                                    is_moderator=True)
                    memories.append(memory)
                    memories = self.updateMemories(memories, [a, self.agents[next_a]])
                else:
                    template_filling = {
                        "taskInstruction": task_instruction,
                        "input": input,
                        "currentDraft": current_draft,
                        "persona": a.persona,
                        "personaDescription": a.persona_description,
                        "sentsMin": feedback_sentences[0],
                        "sentsMax": feedback_sentences[1],
                        "agentMemory": memory_string
                    }
                    memories, agreements = a.participate(use_moderator, memories, unique_id, turn, memory_ids,
                                                            template_filling, extract_all_drafts,
                                                            [a, self.agents[next_a]], agreements)
                unique_id = unique_id + 1

            decision = self.decision_making.decide(agreements, turn)

        return current_draft, turn, agreements

    def discuss_debate(self, task_instruction, input, use_moderator, feedback_sentences=[3, 4], max_turns=None, 
                      context_length=1, include_current_turn_in_memory=False, extract_all_drafts=False, debate_rounds=1):
        decision = None
        turn = 0
        unique_id = 0
        memories = []
        agreements = []

        print(f'''Paradigm: Debate (rounds: {debate_rounds})
                    ┌───┐
          ┌────────►│A 1│◄────────┐
          │         └───┘         │
          │                       │
          │                       │
          │                       │
        ┌─┴─┬──────────────────►┌─┴─┐
        │A 3│                   │A 2│
        └───┘◄──────────────────┴───┘
        ''')

        print("Debate rounds between agents A2, ..., An: " + str(debate_rounds))

        while not decision and (turn < max_turns or max_turns is None):
            turn = turn + 1
            log = "Ongoing. Current turn: " + str(turn)
            if not self.verbose:
                log = "\r" + log + "        "
            print(log, end='')

            # ---- Agent A1
            if use_moderator:
                memory_string, memory_ids, current_draft = self.moderator.getMemoryString(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory
                )

                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.moderator.persona,
                    "personaDescription": self.moderator.persona_description,
                    "agentMemory": memory_string
                }
                res, memory, agreements = self.moderator.draft(unique_id, turn, memory_ids, template_filling,
                                                                extract_all_drafts, agreements, is_moderator=True)
                memories.append(memory)
                memories = self.updateMemories(memories, self.agents)
                unique_id = unique_id + 1
            else:
                memory_string, memory_ids, current_draft = self.panelists[0].getMemoryString(
                    context_length=context_length,
                    turn=turn,
                    include_this_turn=include_current_turn_in_memory
                )
                template_filling = {
                    "taskInstruction": task_instruction,
                    "input": input,
                    "currentDraft": current_draft,
                    "persona": self.panelists[0].persona,
                    "personaDescription": self.panelists[0].persona_description,
                    "agentMemory": memory_string
                }
                res, memory, agreements = self.panelists[0].draft(unique_id, turn, memory_ids, template_filling,
                                                                    extract_all_drafts, agreements, is_moderator=True)
                memories.append(memory)
                memories = self.updateMemories(memories, self.agents)
                unique_id = unique_id + 1

            for r in range(debate_rounds):  # ---- Agents A2, A3, ...
                print("Debate round: " + str(r))
                debate_agreements = []
                for i, a in enumerate(self.agents[1:]):  # similar to relay paradigm
                    memory_string, memory_ids, current_draft = a.getMemoryString(
                        context_length=context_length,
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory
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
                        "agentMemory": memory_string
                    }
                    if r == debate_rounds - 1:  # last debate round
                        agents_to_update = [self.agents[0], a, self.agents[next_a]]
                    else:
                        agents_to_update = [a, self.agents[next_a]]
                    memories, debate_agreements = a.participate(use_moderator, memories, unique_id, turn,
                                                                memory_ids, template_filling, extract_all_drafts,
                                                                agents_to_update, debate_agreements)
                    if len(debate_agreements) > len(self.agents) - 1:
                        debate_agreements = debate_agreements[1 - len(self.agents):]
                    unique_id = unique_id + 1

            agreements = agreements + debate_agreements
            if len(agreements) > len(self.panelists):
                agreements = agreements[-len(self.panelists):]
            decision = self.decision_making.decide(agreements, turn)
    
        return current_draft, turn, agreements

    def discuss(self, task_instruction, input, context, use_moderator, feedback_sentences=[3, 4], paradigm="memory",
                max_turns=None, context_length=1, include_current_turn_in_memory=False, extract_all_drafts=False,
                debate_rounds=1):
        '''
        The routine responsible for the discussion between agents to solve a task.

        The routine is organized as follows:
        1) Create agents with personas
        2) Discuss the problem based on the given paradigm (iteratively check for agreement between agents)
        3) After max turns or agreement reached: return the final result to the task sample
        
        Returns the final response agreed on, the global memory, agent specific memory, turns needed, last agreements of agents
        '''
        if context:
            task_instruction += "\n" + "Context: " + context

        if not self.initAgents(task_instruction, input, use_moderator=use_moderator):
            print("Failed to intialize agents.")
            return None, None, None, None, None, None  # if the LLM failed to initialize the agents, do not discuss

        personas = [a.persona for a in self.agents]
        if len(personas) <= 2:
            print("Only two or less personas were generated. No discussion is executed.")
            return None, None, None, None, None, None  # if the LLM failed to initialize the agents, do not discuss

        self.decision_making = MajorityConsensus(self.panelists)

        print(f'''
Starting discussion...
-------------
Instruction: {task_instruction}
Input: {input}
Feedback sentences: {str(feedback_sentences)}
Maximum turns: {max_turns}
Agents: {str(personas)}
Decision-making: {self.decision_making.__class__.__name__}
-------------''')

        startTime = time.perf_counter()
        if paradigm == "memory":
            current_draft, turn, agreements = self.discuss_memory(task_instruction, input, use_moderator, feedback_sentences,
                                                                  max_turns, context_length, include_current_turn_in_memory, extract_all_drafts)
        elif paradigm == "report": 
            current_draft, turn, agreements = self.discuss_report(task_instruction, input, use_moderator, feedback_sentences,
                                                                  max_turns, context_length, include_current_turn_in_memory, extract_all_drafts)
        elif paradigm == "relay":
            current_draft, turn, agreements = self.discuss_relay(task_instruction, input, use_moderator, feedback_sentences,
                                                                 max_turns, context_length, include_current_turn_in_memory, extract_all_drafts)
        elif paradigm == "debate":
            current_draft, turn, agreements = self.discuss_debate(task_instruction, input, use_moderator, feedback_sentences, max_turns,
                                                                 context_length, include_current_turn_in_memory, extract_all_drafts, debate_rounds)
        discussionTime = timedelta(seconds=time.perf_counter() - startTime).total_seconds()

        globalMem = self.getGlobalMemory()
        agentMems = []
        for a in self.agents:
            agentMems.append(a.getMemory()[0])

        if turn >= max_turns:  # if no agreement was reached
            current_draft = None
        else:
            current_draft = self.chain_extract_result.invoke(
                {
                    "result": current_draft
                })["text"]

        return current_draft, globalMem, agentMems, turn, agreements, discussionTime


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
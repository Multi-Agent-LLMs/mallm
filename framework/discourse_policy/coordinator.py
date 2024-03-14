import sys, os, ast
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
sys.path.append("/beegfs/wahle/github/MALLM")
from framework.agents.agent import *
from framework.agents.panelist import *
from framework.agents.moderator import *
from framework.decision_making.consensus import *
from framework.prompts import coordinator_prompts
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import fire, glob, dbm, re
from langchain_community.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
from setup import *

class Coordinator():
    def __init__(self, use_moderator = True):
        self.personas = None
        self.panelists = []
        self.agents = []
        self.use_moderator = use_moderator
        self.moderator = None
        self.memory_bucket = memory_bucket_dir+"global"
        self.decision_making = None
        self.llm_tokenizer = None
        self.llm = self.create_llm()
        
        if "llama" in self.llm_tokenizer.__class__.__name__.lower():    # use <<SYS>> and [INST] tokens for llama models
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

    def initAgents(self, task_instruction, input, use_moderator, persona_type = "expert"):
        '''
        Instantiates the agents by
        1) identify helpful personas
        2) create agents with the personas
        Gives true if the automatic assignment was successfull.
        Returns bool
        '''

        template_filling = { 
            "task_instruction": task_instruction,
            "input": input
        }

        res = self.chain_identify_personas.invoke(template_filling)["text"]
        
        # TODO: Use grammar to force LLM output in the correct JSON format. Example with llama.ccp: https://til.simonwillison.net/llms/llama-cpp-python-grammars

        # repair dictionary in string if the LLM did mess up the formatting
        if "{" in res and "}" not in res:
            print("Looks like the LLM did not provide a valid dictionary (maybe the last brace is missing?). Trying to repair the dictionary...")
            res = res + "}"

        self.updateGlobalMemory(0, 0, None, None, "persona_identification", res, None, [], template_filling)
        
        personas_string = re.search(r"\{.*?\}", res, re.DOTALL)
        if not personas_string:
            print(f"LLM failed to provide personas in the correct format - Skipping this sample...")
            #personas_string = '''{
            #    "Poet": "A person who studies and creates poetry. The poet is familiar with the rules and formats of poetry and can provide guidance on how to write a poem.",
            #    "Computer Scientist": "A scholar who specializes in the academic study of computer science. The computer scientist is familiar with the concept of a quantum computer and can provide guidance on how to explain it.",
            #    "Ten year old child": "A child with a limited English vocabulary and little knowledge about complicated concepts, such as a quantum computer."
            #    }'''
            #self.personas = ast.literal_eval(personas_string)
            return False
        else:
            personas_string = personas_string.group()
            for i in [0,1]:
                try:
                    self.personas = ast.literal_eval(personas_string)
                except Exception as e:
                    if i == 0:
                        print("Looks like the LLM did not get the formatting quite right. Trying to repair the dictionary string...")
                        personas_string = personas_string.replace("\'\n", "\',\n")
                        personas_string = personas_string.replace('\"\n', '\",\n')
                        print("Repaired string: \n" + str(personas_string))
                        continue
                    elif i == 1:
                        print(f"Failed to parse the string to identify personas: {e} - Skipping this sample...")
                        #personas_string = '''{
                        #"Poet": "A person who studies and creates poetry. The poet is familiar with the rules and formats of poetry and can provide guidance on how to write a poem.",
                        #"Computer Scientist": "A scholar who specializes in the academic study of computer science. The computer scientist is familiar with the concept of a quantum computer and can provide guidance on how to explain it.",
                        #"Ten year old child": "A child with a limited English vocabulary and little knowledge about complicated concepts, such as a quantum computer."
                        #}'''
                        #self.personas = ast.literal_eval(personas_string)
                        return False

        self.panelists = []
        if use_moderator:
            self.moderator = Moderator(0, self.llm, self)  # Agent ID 0 is reserved for the moderator
        for i, p in enumerate(self.personas):
            self.panelists.append(Panelist(i+1, self.llm, p, self.personas[p], self))

        if use_moderator:
            self.agents = [self.moderator] + self.panelists
        else:
            self.agents = self.panelists
        return True

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
        if device == "cpu": # not recommended but useful for developing with no GPU available
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
        #self.llm_tokenizer.pad_token_id = model.config.eos_token_id
        print("Using this tokenizer: " + str(self.llm_tokenizer.__class__.__name__))
        
        pipeline = transformers.pipeline(
            model=model, 
            tokenizer=self.llm_tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            pad_token_id=self.llm_tokenizer.eos_token_id,
            # model parameters
            do_sample=True,
            temperature = 0.9,
            max_new_tokens=512,  # max number of tokens to generate in the output
            min_new_tokens=2,   # always answer something (no empty responses)
            repetition_penalty=1.1,  # without this output begins repeating
        )

        return HuggingFacePipeline(pipeline=pipeline)
    
    def updateGlobalMemory(self, unique_id, turn, agent_id, agent_persona, contribution, text, extracted_draft, memory_ids, prompt_args):
        '''
        Updates the dbm memory with another discussion entry.
        Returns string
        '''
        if extracted_draft:
            extracted_draft = str(extracted_draft).replace('"',"'")
        with dbm.open(self.memory_bucket, 'c') as db:
            db[str(unique_id)] = f'''{{"turn": {turn}, "agent_id": {agent_id}, "persona": "{str(agent_persona).replace('"',"'")}", "prompt_args":{prompt_args}, "contribution": "{contribution}", "memory_ids": {memory_ids}, "text": "{str(text).replace('"',"'")}", "extracted_draft": "{extracted_draft}"}}'''
            print(str(unique_id) + ": " + str(db[str(unique_id)]))   # logging
        self.saveGlobalMemoryToJson()
    
    def getGlobalMemory(self):
        '''
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        '''
        memory = {}
        with dbm.open(self.memory_bucket, 'r') as db:
            for key in db.keys():
                memory[key.decode()] = ast.literal_eval(db[key].decode().replace("\n", "\\n").replace("\t", "\\t"))
        return memory
    
    def saveGlobalMemoryToJson(self):
        '''
        Converts the memory bucket dbm data to json format
        '''
        try:
            with open(self.memory_bucket+".json", "w") as f: 
                json.dump(self.getGlobalMemory(), f)
        except Exception as e:
            print(f"Failed to save agent memory to {self.memory_bucket}: {e}")
            print(self.getGlobalMemory())

    def cleanMemoryBucket(self):
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
        for c in memories:
            for a in agents_to_update:
                a.updateMemory(c["unique_id"], c["turn"], c["id"], c["persona"], c["contribution"], c["text"], c["extracted_draft"], c["memory_ids"], c["template_filling"])
        return []

    def agree(self, res, agreements, is_moderator=False, self_drafted=False):
        if ("agree" in res.lower() and "disagree" not in res.lower()) and (not is_moderator):
            agreements.append(True)
        elif self_drafted:
            agreements.append(True)
        elif not is_moderator:
            agreements.append(False)

        if len(agreements) > len(self.panelists):
            agreements = agreements[len(agreements)-len(self.panelists):]
        return agreements

    def discuss(self, task_instruction, input, use_moderator, feedback_sentences = [3,4], paradigm="memory", max_turns = None, context_length = 1, include_current_turn_in_memory=False, extract_all_drafts=False):
        # 1) Assign agents and personas
        # 2) Create first draft
        # 3) Iterative feedback loop (drafting and checking for decision-making after turn)

        if not self.initAgents(task_instruction, input, use_moderator=use_moderator):
            print("Failed to intialize agents.")
            return None, None, None, None # if the LLM failed to initialize the agents, do not discuss

        personas = [a.persona for a in self.agents]
        if len(personas) <= 1:
            print("Only one or zero personas were generated. No discussion is possible.")
            return None, None, None, None # if the LLM failed to initialize the agents, do not discuss
        
        self.decision_making = MajorityConsensus(self.agents, use_moderator)

        print(f'''
            Starting discussion...
            -------------
            Instruction: {task_instruction}
            Input: {input}
            Feedback sentences: {str(feedback_sentences)}
            Maximum turns: {max_turns}
            Agents: {str(personas)}
            Decision-making: {self.decision_making.__class__.__name__}
            -------------
        ''')
        
        decision = None
        turn = 0
        unique_id = 1
        memories = []
        agreements = []

        if paradigm == "memory": #-----------------------------------------------
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

                if use_moderator:
                    memory_string, memory_ids, current_draft = self.moderator.getMemoryString(
                        context_length=context_length, 
                        turn=turn, 
                        include_this_turn=include_current_turn_in_memory
                        )

                    template_filling = {
                        "task_instruction": task_instruction,
                        "input": input,
                        "current_draft": current_draft,
                        "persona": self.moderator.persona,
                        "persona_description": self.moderator.persona_description,
                        "agent_memory": memory_string
                    }
                    res, memory = self.moderator.draft(unique_id, turn, memory_ids, template_filling, extract_all_drafts)
                    memories.append(memory)
                    memories = self.updateMemories(memories, self.agents)
                    agreements = self.agree(res, agreements, is_moderator=True)
                    unique_id = unique_id + 1

                for p in self.panelists:
                    memory_string, memory_ids, current_draft = p.getMemoryString(
                        context_length=context_length, 
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory
                        )
                    template_filling = {
                        "task_instruction": task_instruction,
                        "input": input,
                        "current_draft": current_draft,
                        "persona": p.persona,
                        "persona_description": p.persona_description,
                        "sents_min": feedback_sentences[0],
                        "sents_max": feedback_sentences[1],
                        "agent_memory": memory_string
                    }

                    memories, agreements = p.participate(use_moderator, memories, agreements, unique_id, turn, memory_ids, template_filling, extract_all_drafts, self.agents)
                    unique_id = unique_id + 1
                
                decision = self.decision_making.decide(agreements, turn)

        elif paradigm == "report": #-----------------------------------------------
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

                # ---- Agent A1
                if use_moderator:
                    memory_string, memory_ids, current_draft = self.moderator.getMemoryString(
                        context_length=context_length, 
                        turn=turn, 
                        include_this_turn=include_current_turn_in_memory
                        )

                    template_filling = {
                        "task_instruction": task_instruction,
                        "input": input,
                        "current_draft": current_draft,
                        "persona": self.moderator.persona,
                        "persona_description": self.moderator.persona_description,
                        "agent_memory": memory_string
                    }
                    res, memory = self.moderator.draft(unique_id, turn, memory_ids, template_filling, extract_all_drafts)
                    memories.append(memory)
                    memories = self.updateMemories(memories, self.agents)
                    agreements = self.agree(res, agreements, is_moderator=True)
                    unique_id = unique_id + 1
                else:
                    memory_string, memory_ids, current_draft = self.panelists[0].getMemoryString(
                        context_length=context_length, 
                        turn=turn, 
                        include_this_turn=include_current_turn_in_memory
                        )
                    template_filling = {
                        "task_instruction": task_instruction,
                        "input": input,
                        "current_draft": current_draft,
                        "persona": self.panelists[0].persona,
                        "persona_description": self.panelists[0].persona_description,
                        "agent_memory": memory_string
                    }
                    res, memory = self.panelists[0].draft(unique_id, turn, memory_ids, template_filling, extract_all_drafts)
                    memories.append(memory)
                    memories = self.updateMemories(memories, self.agents)
                    agreements = self.agree(res, agreements, is_moderator=True, self_drafted=True) # for this paradigm to work, agent 1 has to act as (non-neutral) moderator
                    unique_id = unique_id + 1

                # ---- Agents A2, A3, A4, ...
                for p in self.agents[1:]:
                    memory_string, memory_ids, current_draft = p.getMemoryString(
                        context_length=context_length, 
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory
                        )
                    template_filling = {
                        "task_instruction": task_instruction,
                        "input": input,
                        "current_draft": current_draft,
                        "persona": p.persona,
                        "persona_description": p.persona_description,
                        "sents_min": feedback_sentences[0],
                        "sents_max": feedback_sentences[1],
                        "agent_memory": memory_string
                    }

                    memories, agreements = p.participate(True, memories, agreements, unique_id, turn, memory_ids, template_filling, extract_all_drafts, [self.agents[0], p])
                    unique_id = unique_id + 1

                decision = self.decision_making.decide(agreements, turn)

        elif paradigm == "relay": #-----------------------------------------------
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

                for i, a in enumerate(self.agents):
                    memory_string, memory_ids, current_draft = a.getMemoryString(
                        context_length=context_length, 
                        turn=turn,
                        include_this_turn=include_current_turn_in_memory
                        )
                    next_a = i+1
                    if i == len(self.agents)-1:
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
                        res, memory = self.moderator.draft(unique_id, turn, memory_ids, template_filling, extract_all_drafts)
                        memories.append(memory)
                        memories = self.updateMemories(memories, [a, self.agents[next_a]])
                        agreements = self.agree(res, agreements, is_moderator=True)
                    else:
                        template_filling = {
                            "task_instruction": task_instruction,
                            "input": input,
                            "current_draft": current_draft,
                            "persona": a.persona,
                            "persona_description": a.persona_description,
                            "sents_min": feedback_sentences[0],
                            "sents_max": feedback_sentences[1],
                            "agent_memory": memory_string
                        }
                        memories, agreements = a.participate(use_moderator, memories, agreements, unique_id, turn, memory_ids, template_filling, extract_all_drafts, [a, self.agents[next_a]])
                    unique_id = unique_id + 1
                
                decision = self.decision_making.decide(agreements, turn)

        elif paradigm == "debate": #-----------------------------------------------
            print('''Paradigm: Debate
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
            print("This feature has not been implemented yet.")

        globalMem = self.getGlobalMemory()
        agentMems = []
        for a in self.agents:
            agentMems.append(a.getMemory()[0])

        if turn >= max_turns: # if no agreement was reached
            current_draft = None
        else:
            current_draft = self.chain_extract_result.invoke(
                {
                    "result": current_draft
                })["text"]

        return current_draft, globalMem, agentMems, turn

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)
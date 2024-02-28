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
import fire
import dbm
import ast
import re
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
        self.llm = self.create_llm()
        
        self.chain_identify_personas = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(coordinator_prompts.identify_personas()))
        self.chain_extract_result = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(coordinator_prompts.extract_result()))
    

    def initAgents(self, task_instruction, input, persona_type = "expert", use_moderator = True):
        '''
        Instantiates the agents by
        1) identify helpful personas
        2) create agents with the personas
        Gives true if the automatic assignment was successfull.
        Returns bool
        '''
        res = self.chain_identify_personas.invoke(
            { 
                "task_instruction": task_instruction,
                "input": input
            })["text"]
        
        # repair dictionary in string if the LLM did mess up the formatting
        if "{" in res and "}" not in res:
            print("Looks like the LLM did not provide a valid dictionary (maybe the last brace is missing?). Trying to repair the dictionary...")
            res = res + "}"

        self.updateGlobalMemory(0, 0, None, None, "persona_identification", res, [])
        
        personas_string = re.search(r"\{.*?\}", res, re.DOTALL)
        if not personas_string:
            print(f"LLM failed to provide personas in the correct format - Continue with placeholder personas...")
            personas_string = '''{
                "Poet": "A person who studies and creates poetry. The poet is familiar with the rules and formats of poetry and can provide guidance on how to write a poem.",
                "Computer Scientist": "A scholar who specializes in the academic study of computer science. The computer scientist is familiar with the concept of a quantum computer and can provide guidance on how to explain it.",
                "Ten year old child": "A child with a limited English vocabulary and little knowledge about complicated concepts, such as a quantum computer."
                }'''
            self.personas = ast.literal_eval(personas_string)
            #return False
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
                        print(f"Failed to parse the string to identify personas: {e} - Continue with placeholder personas...")
                        personas_string = '''{
                        "Poet": "A person who studies and creates poetry. The poet is familiar with the rules and formats of poetry and can provide guidance on how to write a poem.",
                        "Computer Scientist": "A scholar who specializes in the academic study of computer science. The computer scientist is familiar with the concept of a quantum computer and can provide guidance on how to explain it.",
                        "Ten year old child": "A child with a limited English vocabulary and little knowledge about complicated concepts, such as a quantum computer."
                        }'''
                        self.personas = ast.literal_eval(personas_string)
                        continue

        self.panelists = []
        if use_moderator:
            self.moderator = Moderator(0, self.llm, self)  # Agent ID 0 is reserved for the moderator
        for i, p in enumerate(self.personas):
            self.panelists.append(Panelist(i+1, self.llm, p, self.personas[p], self))

        if use_moderator:
            self.agents = self.panelists + [self.moderator]
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

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            ckpt_dir
        )
        print("Using this tokenizer: " + str(tokenizer.__class__.__name__))
        
        pipeline = transformers.pipeline(
            model=model, 
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            pad_token_id=tokenizer.eos_token_id,
            # model parameters
            do_sample=True,
            temperature = 0.9,
            max_new_tokens=512,  # max number of tokens to generate in the output
            min_new_tokens=2,   # always answer something (no empty responses)
            repetition_penalty=1.1,  # without this output begins repeating
        )

        return HuggingFacePipeline(pipeline=pipeline)
    
    def updateGlobalMemory(self, unique_id, turn, agent_id, agent_persona, contribution, text, memory_ids):
        '''
        Updates the dbm memory with another discussion entry.
        Returns string
        '''
        with dbm.open(self.memory_bucket, 'c') as db:
            db[str(unique_id)] = f'''{{"turn": {turn}, "agent_id": {agent_id}, "agent_persona": "{str(agent_persona).replace('"',"'")}", "contribution": "{contribution}", "memory_ids": {memory_ids}, "text": "{str(text).replace('"',"'")}"}}'''
            print(db[str(unique_id)])   # logging
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

    def discuss(self, task_instruction, input, decision_threshold, use_moderator, avg_feedback_length = 3, paradigm="memory", max_turns = None):
        # 1) Assign agents and personas
        # 2) Create first draft
        # 3) Iterative feedback loop (drafting and checking for decision-making after turn)

        if not self.initAgents(task_instruction, input, use_moderator=use_moderator):
            print("Failed to intialize agents.")
            return None # if the LLM failed to initialize the agents, do not discuss
        self.decision_making = MajorityConsensus(self.agents, use_moderator)

        personas = [a.persona for a in self.agents]
        print(f'''
            Starting discussion...
            -------------
            Instruction: {task_instruction}
            Input: {input}
            Average feedback length: {str(avg_feedback_length)}
            Maximum turns: {max_turns}
            Agents: {str(personas)}
            Decision-making: {self.decision_making.__class__.__name__}
            -------------
        ''')
        
        decision = None
        current_draft = None
        turn = 0
        unique_id = 1
        feedbacks = []
        feedback_ids = []
        agreements = []

        #for p in self.panelists:
        #    feedbacks.append(p.brainstorm(unique_id, turn, task_instruction, avg_feedback_length, self.agents, input))
        #    unique_id = unique_id + 1

        if paradigm == "memory":
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
                for p in self.panelists:
                    res = p.improve(
                        unique_id = unique_id, 
                        turn = turn, 
                        task_instruction = task_instruction, 
                        current_draft = current_draft, 
                        feedback_sentences = avg_feedback_length, 
                        agents_to_update = self.agents,
                        input = input,
                        feedbacks = feedbacks,
                        feedback_ids = feedback_ids,
                        context_length = len(self.panelists)-1
                        )
                    feedbacks.append(p.persona + ": " + res)
                    feedback_ids.append(unique_id)

                    if "AGREE" in res and "DISAGREE" not in res:
                        agreements.append(True)
                    else:
                        current_draft = self.chain_extract_result.invoke(
                            {
                                "result": res
                            })["text"]
                        agreements.append(False)
                    if len(agreements) > len(self.panelists):
                        agreements = agreements[len(agreements)-len(self.panelists):]   # only keep recent agreements
                    unique_id = unique_id + 1
                
                decision = self.decision_making.decide(agreements, turn)

        elif paradigm == "memory_BACKUP":
    
            while (not decision and max_turns is None) or (max_turns is not None and turn <= max_turns):
                if self.use_moderator:
                    current_draft = self.moderator.createDraft(unique_id, turn, task_instruction, input, self.agents, context_length=None)
                else:
                    current_draft = self.panelists[0].createDraft(unique_id, turn, task_instruction, input, self.agents, context_length=None)
                unique_id = unique_id + 1
                turn = turn + 1
                agreements = []

                for p in self.panelists:
                    feedback, agreement, agreement_reason = p.generateFeedback(
                        unique_id = unique_id, 
                        turn = turn, 
                        task_instruction = task_instruction, 
                        current_draft = current_draft, 
                        feedback_sentences = avg_feedback_length, 
                        agents_to_update = self.agents, 
                        input = input
                        )
                    feedbacks.append(feedback)
                    agreements.append(agreement)
                    unique_id = unique_id + 1
                
                if max_turns is None:
                    decision = self.decision_making.decide(agreements)

        elif paradigm == "report":
            '''
                        ┌───┐
                        │A 1│
                ┌──────►└┼─┼┘◄──────┐
                │        │ │        │
                │        │ │        │
                │        │ │        │
            ┌───┼────────┘ └───────►├───┐
            │A 3│                   │A 2│
            └───┘                   └───┘
            '''
            print("This feature has not been implemented yet.")

        elif paradigm == "relay":
            '''
                        ┌───┐
              ┌────────►│A 1│─────────┐
              │         └───┘         │
              │                       │
              │                       │
              │                       ▼
            ┌─┴─┐                   ┌───┐
            │A 3│◄──────────────────┤A 2│
            └───┘                   └───┘
            '''
            print("This feature has not been implemented yet.")
        elif paradigm == "debate":
            '''
                        ┌───┐
              ┌────────►│A 1│◄────────┐
              │         └───┘         │
              │                       │
              │                       │
              │                       │
            ┌─┴─┬──────────────────►┌─┴─┐
            │A 3│                   │A 2│
            └───┘◄──────────────────┴───┘
            '''
            print("This feature has not been implemented yet.")

        globalMem = self.getGlobalMemory()
        agentMems = []
        for a in self.agents:
            agentMems.append(a.getMemory())

        if turn >= max_turns and max_turns is not None: # if no agreement was reached
            current_draft = None

        return current_draft, globalMem, agentMems, turn

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)
import sys, os, ast
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
sys.path.append("/beegfs/wahle/github/MALLM")
#sys.path.append("/beegfs/wahle/github/MALLM/framework")
#sys.path.append("/beegfs/wahle/github/MALLM/framework/discourse_policy")
#sys.path.append("/beegfs/wahle/github/MALLM/framework/decision_making")
#sys.path.append("/beegfs/wahle/github/MALLM/framework/agents")
#sys.path.append("/beegfs/wahle/github/MALLM/framework/models")
#sys.path.append("/beegfs/wahle/github/MALLM/models/llama")
from framework.agents.agent import *
from framework.agents.panelist import *
from framework.agents.moderator import *
from framework.decision_making.consensus import *
from framework.prompts import coordinator_prompts
import fire
import dbm
import ast
import re
from langchain_community.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import torch
import transformers
from setup import *

class Coordinator():
    def __init__(self, task_name, task_description, decision_threshold, use_moderator = True):
        self.personas = None
        self.panelists = []
        self.agents = []
        self.use_moderator = use_moderator
        self.moderator = None
        self.memory_bucket = memory_bucket_dir+"global"
        self.decision_making = None
        self.llm = self.create_llm()
    

    def initAgents(self, task_name, task_description, task_instruction, source_text, persona_type = "expert", use_moderator = True):
        res = self.llm.invoke(self.updateGlobalMemory(0, 0, 0, None, coordinator_prompts.identify_personas(task_instruction, source_text)))
        print(res)
        personas_string = re.search(r"\{.*?\}", res, re.DOTALL)
        if not personas_string:
            return False
        self.personas = ast.literal_eval(personas_string.group())
        print(self.personas)
        if use_moderator:
            self.moderator = Moderator(0, self.llm, self)  # Agent ID 0 is reserved for the moderator
        for i, p in enumerate(self.personas):
            self.panelists.append(Panelist(i+1, self.llm, p, self))

        if use_moderator:
            self.agents = self.panelists + [self.moderator]
        else:
            self.agents = self.panelists
        return True

    def create_llm(self):
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        print(f"Running on device: {device}")
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        model_config = transformers.AutoConfig.from_pretrained(
            ckpt_dir_llama2
        )
        if device == "cpu": # not recommended but useful for developing with no GPU available
            model = transformers.AutoModelForCausalLM.from_pretrained(
                ckpt_dir_llama2,
                trust_remote_code=True,
                config=model_config,
                offload_folder="offload",
                device_map='auto'
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                ckpt_dir_llama2,
                trust_remote_code=True,
                config=model_config,
                quantization_config=bnb_config,
                device_map='auto'
            )
        model.eval()
        print(f"Model loaded on {device}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            ckpt_dir_llama2
        )
        
        pipeline = transformers.pipeline(
            model=model, 
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            pad_token_id=tokenizer.eos_token_id,
            # model parameters
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.1,  # without this output begins repeating
        )

        return HuggingFacePipeline(pipeline=pipeline)
    
    def updateGlobalMemory(self, unique_id, turn, agent_id, agent_persona, text):
        with dbm.open(self.memory_bucket, 'c') as db:
            db[str(unique_id)] = f'''{{"turn": {turn}, "agent_id": {agent_id}, "agent_persona": "{agent_persona}", "text": "{text}"}}'''
        self.saveGlobalMemoryToJson()
        return text
    
    def getGlobalMemory(self):
        '''
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        '''
        memory = {}
        with dbm.open(self.memory_bucket, 'r') as db:
            for key in db.keys():
                print(ast.literal_eval(db[key].decode()))
                memory[key.decode()] = ast.literal_eval(db[key].decode())
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

    def discuss(self, task_name, task_description, task_instruction, source_text, decision_threshold, use_moderator, avg_feedback_length = 3, paradigm="memory"):
        # 1) Assign agents and personas
        # 2) Create first draft
        # 3) Iterative feedback loop (drafting and checking for decision-making after turn)

        if not self.initAgents(task_name, task_description, task_instruction, source_text):
            print("Failed to intialize agents.")
            return None # if the LLM failed to initialize the agents, do not discuss
        self.decision_making = Consensus(self.agents, decision_threshold, use_moderator)
        decision = None
        current_draft = None
        turn = 0
        unique_id = 1
        feedbacks = []

        for p in self.panelists:
            feedbacks.append(p.brainstorm(unique_id, turn, task_name, task_description, avg_feedback_length, self.agents, source_text))
            unique_id = unique_id + 1

        if paradigm == "memory":
            '''
                        ┌───┐
                        │A 1│
                        ├───┘
                        │   ▲
                        │   │
                        ▼   │
            ┌───┬──────►┌───┤◄──────┬───┐
            │A 3│       │MEM│       │A 2│
            └───┘◄──────┴───┴──────►└───┘
            '''

            while not decision or turn < 40:
                if self.use_moderator:
                    current_draft = self.moderator.createDraft(unique_id, turn, task_name, task_description, source_text, self.agents, context_length=None, use_moderator=True)
                else:
                    current_draft = self.panelists[0].createDraft(unique_id, turn, task_name, task_description, source_text, self.agents, context_length=None, use_moderator=False)
                unique_id = unique_id + 1
                turn = turn + 1
                agreements = []
                # TODO
                for p in self.panelists:
                    feedback, agreement = p.generateFeedback(unique_id, turn, task_name, task_description, current_draft, avg_feedback_length, self.agents, source_text)
                    feedbacks.append(feedback)
                    agreements.append(agreement)
                    unique_id = unique_id + 1
                
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
        return current_draft



def main(task_name = "Paraphrasing", task_description = "In paraphrasing you need to modify the source text while not changing its meaning.", decision_threshold = None, use_moderator = True):
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
    
    coordinator = Coordinator(task_name, task_description, decision_threshold, use_moderator)
    source_text = '''The 24-year-old spent six seasons with the north London side and has previously spent time playing in the second tier with Bedford Blues. The Exiles have not disclosed the length of the former England under-20 international's contract. "Ben is a great acquisition," director of rugby Nick Kennedy said. "He has Championship experience which will be very useful as we gear up for what will be a very competitive campaign."'''
    task_instruction = "Summarize this text:"
    answer = coordinator.discuss(task_name, task_description, task_instruction, source_text, decision_threshold, use_moderator, avg_feedback_length = 3, paradigm="memory")
    print("FINAL ANSWER: ")
    print(answer)

if __name__ == "__main__":
    fire.Fire(main)
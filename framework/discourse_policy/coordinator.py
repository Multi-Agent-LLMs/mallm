import sys, os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
sys.path.append("/beegfs/wahle/github/MALLM")
sys.path.append("/beegfs/wahle/github/MALLM/framework")
sys.path.append("/beegfs/wahle/github/MALLM/framework/discourse_policy")
sys.path.append("/beegfs/wahle/github/MALLM/framework/decision_making")
sys.path.append("/beegfs/wahle/github/MALLM/agents")
sys.path.append("/beegfs/wahle/github/MALLM/models")
sys.path.append("/beegfs/wahle/github/MALLM/models/llama")
from framework.agents.agent import *
from framework.prompts import coordinator_prompts
import fire
from langchain_community.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import torch
import transformers
from setup import *

class Coordinator():
    def __init__(self, personas, task_name, task_description):
        self.personas = personas
        self.agents = []
        self.llm = self.create_role_assignment_llm()
        self.initAgents(task_name, task_description)
    

    def initAgents(self, task_name, task_description):
        output = self.llm.invoke(coordinator_prompts.coordinate_personas("Question-Answering", "Question-Answering needs to returns a factually correct answer to a source question.", "expert", "Who invented language models?"))
        print(output)

        for i, p in enumerate(self.personas):
            self.agents.append(Agent(i, "placeholder model", p))

    def create_role_assignment_llm(self):
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
            # model parameters
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

        return HuggingFacePipeline(pipeline=pipeline)

def main():
    Coordinator(["developer", "business manager", "ai researcher"], "Paraphrasing", "Please paraphrase... placeholder prompt")

if __name__ == "__main__":
    fire.Fire(main)
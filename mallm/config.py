OPENAI_API_KEY = None
#ckpt_dir = "/beegfs/wahle/llama-2-weights/llama-2-70b-chat-hf"
#ckpt_dir = "microsoft/DialoGPT-medium"
#ckpt_dir = "microsoft/GODEL-v1_1-base-seq2seq"
#ckpt_dir = "PY007/TinyLlama-1.1B-step-50K-105b"
ckpt_dir = "berkeley-nest/Starling-LM-7B-alpha"

memory_bucket_dir = "./experiments/memory_bucket/"

PARADIGMS = ["memory", "report", "relay", "debate"]

sample_size = 30    # how many samples to extract from the dataset. Choose 30 because of the central limit theorem

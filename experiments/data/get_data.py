from datasets import load_dataset
import sys, os
sys.path.append("/beegfs/wahle/github/MALLM")
from setup import *
from experiments.data.etpc import *
import urllib.request
import json, random, uuid

# WMT19 (Translation)
#print("Downloading WMT19...")
#dataset = load_dataset('wmt19', "de-en")
#dataset.save_to_disk("./wmt19.hf")
#dataset = dataset["validation"].shuffle(seed=42).select(range(sample_size))
#dataset.to_json("experiments/data/wmt19.json") 

print("Downloading Europarl...")
# Europarl (german to english) (Translation)
dataset = load_dataset("Helsinki-NLP/europarl", "de-en")
dataset.save_to_disk("./europarl.hf")
dataset = dataset["train"].shuffle(seed=42).select(range(sample_size))
json_str = ""
for s in dataset.select(range(sample_size)).iter(batch_size=1):
    json_str += f'''{{ "id":"{str(uuid.uuid4())}", "input":{json.dumps(s['translation'][0]['de'])}, "context": null, "references": {json.dumps(s['translation'][0]['en'])}, "personas": null }}\n'''
with open("experiments/data/europarl.json", 'w') as file:
    file.write(json_str)


print("Downloading SQuAD 2.0...")
# SQuAD 2.0 (QA)
dataset = load_dataset("rajpurkar/squad_v2")
dataset.save_to_disk("./squad_v2.hf")
dataset = dataset["validation"].shuffle(seed=42).select(range(sample_size))
json_str = ""
for s in dataset.select(range(sample_size)).iter(batch_size=1):
    json_str += f'''{{ "id":"{s["id"][0]}", "input":{json.dumps(s['question'][0])}, "context": {json.dumps(s['context'][0])}, "references": {json.dumps(s['answers'][0]["text"])}, "personas": null }}\n'''
with open("experiments/data/squad_v2.json", 'w') as file:
    file.write(json_str)

print("Downloading Simple Ethical Questions...")
# Simple Ethical Questions (QA)
urllib.request.urlretrieve("https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/simple_ethical_questions/task.json", "experiments/data/simple_ethical_questions.json")
dataset = json.loads(open("experiments/data/simple_ethical_questions.json").read())["examples"]
random.shuffle(dataset)
json_str = ""
for s in dataset[sample_size:]:
    ref = [k for k, v in s["target_scores"].items() if v == 1]
    print(ref)
    json_str += f'''{{ "id":"{str(uuid.uuid4())}", "input":{json.dumps(s['input'])}, "context": null, "references": {json.dumps(ref[0])}, "personas": null }}\n'''
with open("experiments/data/simple_ethical_questions.json", 'w') as file:
    file.write(json_str)

print("Downloading Strategy QA...")
# Strategy QA (QA)
urllib.request.urlretrieve("https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/strategyqa/task.json", "experiments/data/strategyqa.json")
dataset = json.loads(open("experiments/data/strategyqa.json").read())["examples"]
random.shuffle(dataset)
json_str = ""
for s in dataset[sample_size:]:
    json_str += f'''{{ "id":"{str(uuid.uuid4())}", "input":{json.dumps(s['input'])}, "context": null, "references": {json.dumps(s['target'])}, "personas": null }}\n'''
with open("experiments/data/strategyqa.json", 'w') as file:
    file.write(json_str)

print("Downloading XSUM...")
# XSUM (Summarization)
dataset = load_dataset('GEM/xsum')
dataset.save_to_disk("./xsum.hf")
dataset = dataset["test"].shuffle(seed=42).select(range(sample_size))
json_str = ""
for s in dataset.select(range(sample_size)).iter(batch_size=1):
    json_str += f'''{{ "id":"{s["xsum_id"][0]}", "input":{json.dumps(s['document'][0])}, "context": null, "references": {json.dumps(s['references'][0])}, "personas": null }}\n'''
with open("experiments/data/xsum.json", 'w') as file:
    file.write(json_str)

print("Downloading ETPC...")
# ETPC (Paraphrasing)
dataset = load_dataset("jpwahle/etpc")
dataset.save_to_disk("./etpc.hf")
dataset = dataset["train"].shuffle(seed=42).select(range(sample_size))
json_str = ""
for s in dataset.select(range(sample_size)).iter(batch_size=1):
    json_str += f'''{{ "id":"{s["idx"][0]}", "input":{json.dumps(s['sentence1'][0])}, "context": null, "references": {json.dumps(s['sentence2'][0])}, "personas": null }}\n'''
with open("experiments/data/etpc.json", 'w') as file:
    file.write(json_str)

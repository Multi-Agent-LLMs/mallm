"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022

Via

https://github.com/openai/simple-evals/blob/main/gpqa_eval.py
"""




ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"

import re
import json

out_file = "test_out.json"

with open(out_file, "r") as f:
    data = json.load(f)

total_score = 0

for question in data:
    answer = question["answer"]

    reference = question["references"][0]

    #print("Question:", question["input"])
    print("\nAnswer:", answer)
    print("Reference:", reference)

    match = re.search(ANSWER_PATTERN_MULTICHOICE, answer)
    print("Extracted answer:", match.group(1) if match else None)
    extracted_answer = match.group(1) if match else None
    score = 1 if extracted_answer == reference else 0
    total_score += score

print("Total score:", total_score)
print("Total questions:", len(data))
print("Accuracy:", total_score / len(data))

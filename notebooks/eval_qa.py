"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022

Via

https://github.com/openai/simple-evals/blob/main/gpqa_eval.py
"""

import sys


ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"

import re
import json

out_file = sys.argv[1]

with open(out_file, "r") as f:
    data = json.load(f)

total_score = 0
total_questions = 0

for question in data:
    answer = question["answer"]

    if answer is None:
        print("\nNo answer found in question")
        continue

    reference = question["references"][0]

    # print("Question:", question["input"])
    print('\nAnswer: "', answer, '"', sep="")

    match = re.search(ANSWER_PATTERN_MULTICHOICE, answer)
    # Backup
    if not match:
        match = re.search(r"(?i)([A-D])(\W|$)", answer)

    if not match:
        print("No match found in answer.")
        continue

    print("Extracted answer:", match.group(1) if match else None)
    extracted_answer = match.group(1) if match else None
    print("Reference:", reference)
    score = 1 if extracted_answer == reference else 0
    total_score += score
    total_questions += 1

print("\n\nTotal score:", total_score)
print("Total questions:", total_questions)
print("Accuracy:", total_score / total_questions)

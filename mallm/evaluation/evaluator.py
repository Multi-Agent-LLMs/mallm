import json
from typing import Any, Optional

import fire
from tqdm import tqdm

from mallm.evaluation.metrics.bertscore import BERTScore
from mallm.evaluation.metrics.bleu import BLEU
from mallm.evaluation.metrics.meteor import METEOR
from mallm.evaluation.metrics.rouge import ROUGE

# ANSWER_PATTERN_MULTICHOICE = re.compile(r"(?i)Answer\s*:\s*([A-D])")

ALL_METRICS = [BLEU(), ROUGE(), BERTScore(), METEOR()]


class Evaluator:
    def __init__(
        self,
        input_file_path: str,
        output_file_path: str,
        metrics: Optional[list[str]] = None,
    ) -> None:
        if metrics is None:
            metrics = ["bleu"]
        self.output_file_path = output_file_path
        with open(input_file_path) as file:
            self.data = json.load(file)

        metrics = [m.lower() for m in metrics]

        self.metrics = [
            metric_class
            for metric_class in ALL_METRICS
            if metric_class.name.lower() in metrics
        ]

        print("Metrics to calculate: " + str([m.name for m in self.metrics]))

        if not self.metrics:
            raise ValueError(f"No metrics found for {metrics}")

    def calculate_scores(self, answer: str, references: list[str]) -> dict[str, Any]:
        # Tokenize the answer and references
        scores: dict[str, Any] = {}
        for metric in self.metrics:
            scores = {**scores, **metric.evaluate(answer, references)}
        return scores

    def add_scores(self) -> None:
        for item in tqdm(self.data):
            answer = item.get("answer", "")
            references = item.get("references", [])
            if answer and references:
                score = self.calculate_scores(answer, references)
                item["scores"] = score

    def save_json(self) -> None:
        with open(self.output_file_path, "w") as file:
            json.dump(self.data, file, indent=4)

    def process(self) -> None:
        self.add_scores()
        self.save_json()


def main(input_file_path: str, output_file_path: str, metrics: list[str]) -> None:
    evaluator = Evaluator(input_file_path, output_file_path, metrics)
    evaluator.process()
    print(f"Scores added and saved to {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)

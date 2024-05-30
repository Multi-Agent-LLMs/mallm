import json
from mallm.evaluation.metrics.bleu import BLEU
from mallm.evaluation.metrics.rouge import ROUGE
from mallm.evaluation.metrics.bertscore import BERTScore
from mallm.evaluation.metrics.meteor import METEOR
import fire
from tqdm import tqdm
from typing import Any


class Evaluator:
    def __init__(
        self, input_file_path: str, output_file_path: str, metrics: list[str] = ["bleu"]
    ) -> None:
        self.output_file_path = output_file_path
        with open(input_file_path, "r") as file:
            self.data = json.load(file)

        all_metrics = [BLEU(), ROUGE(), BERTScore(), METEOR()]
        metrics = [m.lower() for m in metrics]

        self.metrics = []
        for metric in all_metrics:
            if metric.get_metric_name().lower() in metrics:
                self.metrics.append(metric)

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
    print("Metrics to calculate: " + str(metrics))
    evaluator = Evaluator(input_file_path, output_file_path, metrics)
    evaluator.process()
    print(f"Scores added and saved to {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)

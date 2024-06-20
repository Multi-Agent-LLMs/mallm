import json
import logging
from typing import Any, Optional

import fire
from tqdm import tqdm

import mallm.scheduler  # noqa
from mallm.evaluation.metrics.bertscore import BERTScore
from mallm.evaluation.metrics.bleu import BLEU
from mallm.evaluation.metrics.meteor import METEOR
from mallm.evaluation.metrics.qa import MultiChoiceBoolean
from mallm.evaluation.metrics.rouge import ROUGE

ALL_METRICS = [BERTScore(), BLEU(), METEOR(), MultiChoiceBoolean(), ROUGE()]

logger = logging.getLogger("mallm")


class Evaluator:
    def __init__(
        self,
        input_file_path: str,
        output_file_path: Optional[str] = None,
        metrics: Optional[list[str]] = None,
    ) -> None:
        if metrics is None:
            metrics = ["multichoice"]
        if output_file_path is None:
            output_file_path = input_file_path.replace(".json", "_eval.json")
        self.output_file_path = output_file_path
        with open(input_file_path) as file:
            self.data = json.load(file)

        metrics = [m.lower() for m in metrics]

        self.metrics = [
            metric_class
            for metric_class in ALL_METRICS
            if metric_class.name.lower() in metrics
        ]

        if len(self.metrics) != len(metrics):
            logger.warning(f"Some metrics not found in {metrics}")

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

    def calculate_statistics(self) -> None:
        # For each numeric metric, calculate the average and standard deviation
        reported_metrics = self.data[0]["scores"].keys()

        for metric in reported_metrics:
            logger.info(f"Statistics for: {metric.upper()}")
            scores = [item.get("scores", {}).get(metric, None) for item in self.data]
            scores = [score for score in scores if score is not None]
            if not scores:
                logger.warning(f"No scores found for {metric}")
                continue

            if not all(isinstance(score, (int, float)) for score in scores):
                logger.info(f"Scores for {metric} are not numeric")
                continue

            average_score = sum(scores) / len(scores)
            std_dev_score = (
                sum((score - average_score) ** 2 for score in scores) / len(scores)
            ) ** 0.5

            logger.info(f"Data size: {len(self.data)}")
            logger.info(f"Sample size: {len(scores)}")
            logger.info(f"Average score: {average_score:.2%}")
            logger.info(f"Standard deviation: {std_dev_score:.3f}")

    def save_json(self) -> None:
        with open(self.output_file_path, "w") as file:
            json.dump(self.data, file, indent=4)

    def process(self) -> None:
        self.add_scores()
        self.calculate_statistics()
        self.save_json()
        logger.info(f"Scores saved to {self.output_file_path}")


def main(
    input_file_path: str,
    output_file_path: Optional[str] = None,
    metrics: Optional[list[str]] = None,
) -> None:
    evaluator = Evaluator(input_file_path, output_file_path, metrics)
    evaluator.process()


if __name__ == "__main__":
    fire.Fire(main)

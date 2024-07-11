import json
import logging
from pathlib import Path
from typing import Any, Optional

import fire
from tqdm import tqdm

import mallm.scheduler  # noqa
from mallm.evaluation.metrics.bertscore import BERTScore
from mallm.evaluation.metrics.bleu import BLEU
from mallm.evaluation.metrics.meteor import METEOR
from mallm.evaluation.metrics.qa import (
    AnswerabilityBoolean,
    MultiChoiceBoolean,
    SquadScore,
)
from mallm.evaluation.metrics.rouge import ROUGE

ALL_METRICS = [
    AnswerabilityBoolean(),
    BERTScore(),
    BLEU(),
    METEOR(),
    MultiChoiceBoolean(),
    ROUGE(),
    SquadScore(),
]

logger = logging.getLogger("mallm")


class Evaluator:
    def __init__(
        self,
        input_file_path: str,
        output_file_path: Optional[str] = None,
        metrics: Optional[list[str]] = None,
    ) -> None:
        self.input_file_path = Path(input_file_path)
        self.output_file_path = (
            Path(output_file_path)
            if output_file_path
            else self.input_file_path.with_suffix(".stats.json")
        )
        self.data = self._load_data()
        self.metrics = self._initialize_metrics(metrics)

    def _load_data(self) -> list[dict[str, Any]]:
        with open(self.input_file_path) as file:
            data: list[dict[str, Any]] = json.load(file)
            return data

    @staticmethod
    def _initialize_metrics(metrics: Optional[list[str]]) -> list[Any]:
        if metrics is None:
            metrics = ["multichoice"]
        metrics = [m.lower() for m in metrics]
        selected_metrics = [
            metric_class
            for metric_class in ALL_METRICS
            if metric_class.name.lower() in metrics
        ]
        if len(selected_metrics) != len(metrics):
            logger.warning(f"Some metrics not found in {metrics}")
        if not selected_metrics:
            raise ValueError(f"No metrics found for {metrics}")
        logger.info(f"Metrics to calculate: {[m.name for m in selected_metrics]}")
        return selected_metrics

    def calculate_scores(self, answer: str, references: list[str]) -> dict[str, Any]:
        if references:
            metrics = self.metrics
        elif any(metric.name == "answerability" for metric in self.metrics):
            metrics = [AnswerabilityBoolean()]
        elif any(metric.name == "squad" for metric in self.metrics):
            metrics = [SquadScore()]
        else:
            logger.warning("No metrics to evaluate.")
            return {}

        return {
            k: v
            for metric in metrics
            for k, v in metric.evaluate(answer, references).items()
        }

    def add_scores(self) -> None:
        for item in tqdm(self.data, desc="Calculating scores"):
            answer = item.get("answer", "")
            references = item.get("references", [])
            if answer:
                item["scores"] = self.calculate_scores(answer, references)

    def calculate_statistics(self) -> dict[str, Any]:
        reported_metrics = set()
        for item in self.data:
            if "scores" in item:
                reported_metrics.update(item["scores"].keys())

        if not reported_metrics:
            logger.error("No elements with scores found in the data.")
            raise Exception("No elements with scores found in the data.")

        logger.info(f"Reported metrics: {reported_metrics}")

        stats = {}
        for metric in reported_metrics:
            logger.info(f"-> Statistics for: {metric.upper()}, {self.output_file_path}")
            scores = [item.get("scores", {}).get(metric) for item in self.data]
            scores = [score for score in scores if isinstance(score, (int, float))]

            if not scores:
                logger.warning(f"No numeric scores found for {metric}")
                continue

            average_score = sum(scores) / len(scores)
            std_dev_score = (
                sum((score - average_score) ** 2 for score in scores) / len(scores)
            ) ** 0.5

            stats[metric] = {
                "data_size": len(self.data),
                "sample_size": len(scores),
                "scores": scores,
                "average_score": round(average_score, 4),
                "std_dev_score": round(std_dev_score, 4),
            }
            logger.info(f"Stats: {stats[metric]}")

        return stats

    def save_results(self, stats: dict[str, Any]) -> None:
        with open(self.output_file_path, "w") as file:
            json.dump(stats, file, indent=4)
        logger.info(f"Statistics saved to {self.output_file_path}")

    def process(self) -> None:
        self.add_scores()
        stats = self.calculate_statistics()
        self.save_results(stats)


def batch_process_folder(
    input_folder: str,
    output_folder: Optional[str] = None,
    metrics: Optional[list[str]] = None,
) -> None:
    input_path = Path(input_folder)
    output_path = Path(output_folder) if output_folder else input_path
    print(f"Processing files in {input_path} and saving results to {output_path}")
    print(f"Metrics to calculate: {metrics}")
    print(f"Files to process: {len(list(input_path.glob('*.json')))}")
    files = list(input_path.glob("*.json"))
    for file in files:
        if file.stem.endswith(("_eval", "_stats")):
            continue

        output_file = output_path / file.name.replace(".json", "_eval.json")
        logger.info(f"Processing {file}")

        evaluator = Evaluator(str(file), str(output_file), metrics)
        evaluator.process()

    logger.info("Batch processing completed.")


def main(
    input_path: str,
    output_path: Optional[str] = None,
    metrics: Optional[list[str]] = None,
    batch: bool = False,
) -> None:
    if batch:
        batch_process_folder(input_path, output_path, metrics)
    else:
        evaluator = Evaluator(input_path, output_path, metrics)
        evaluator.process()


if __name__ == "__main__":
    fire.Fire(main)

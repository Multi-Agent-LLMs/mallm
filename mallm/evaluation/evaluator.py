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
        output_dir_path: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        extensive: bool = False,
    ) -> None:
        self.input_file_path = Path(input_file_path)
        if not self.input_file_path.exists() or not self.input_file_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_file_path}")
        output_file_path = (
            Path(output_dir_path) / self.input_file_path.name
            if output_dir_path
            else self.input_file_path
        )
        self.stats_file_path = output_file_path.with_suffix("-stats.json")
        self.eval_file_path = output_file_path.with_suffix("-eval.json")
        self.data = self._load_data()
        self.metrics = self._initialize_metrics(metrics)
        self.extensive = extensive

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

    def add_scores_extensive(self) -> None:
        for item in tqdm(self.data):
            for mem in item.get("globalMemory", []):
                solution = mem.get("solution", "")
                references = item.get("references", [])
                if solution:
                    score = self.calculate_scores(solution, references)
                    mem["scores"] = score

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
            logger.info(f"-> Statistics for: {metric.upper()}, {self.stats_file_path}")
            scores = [item.get("scores", {}).get(metric) for item in self.data]
            scores = [score for score in scores if isinstance(score, (int, float))]

            if not scores:
                logger.warning(f"No numeric scores found for {metric}")
                continue

            average_score = sum(scores) / len(scores)
            if len(scores) > 1:
                std_dev_score = (
                    sum((score - average_score) ** 2 for score in scores)
                    / (len(scores) - 1)
                ) ** 0.5
            else:
                std_dev_score = 0

            avg_scores_per_turn = {}
            if self.extensive:
                for item in self.data:
                    for mem in item.get("globalMemory", []):
                        turn = mem.get("turn", 0)
                        if turn not in avg_scores_per_turn:
                            avg_scores_per_turn[turn] = float(0)
                        avg_scores_per_turn[turn] += mem.get("scores", {}).get(
                            metric, 0
                        )

                max_turns = max(item.get("turns", 0) for item in self.data)
                for turn in range(max_turns + 1)[1:]:
                    avg_scores_per_turn[turn] /= sum(
                        1
                        for item in self.data
                        for mem in item.get("globalMemory", [])
                        if mem.get("turn", 0) == turn
                        and metric in mem.get("scores", {})
                    )
                    avg_scores_per_turn[turn] = round(avg_scores_per_turn[turn], 4)

            stats[metric] = {
                "data_size": len(self.data),
                "sample_size": len(scores),
                "scores": scores,
                "average_score": round(average_score, 4),
                "std_dev_score": round(std_dev_score, 4),
                "average_scores_per_turn": avg_scores_per_turn,
            }
            logger.debug(f"Stats: {stats[metric]}")

        return stats

    def save_results(self, stats: dict[str, Any]) -> None:
        with open(self.stats_file_path, "w") as file:
            json.dump(stats, file, indent=4)
        with open(self.eval_file_path, "w") as file:
            json.dump(self.data, file, indent=4)
        logger.info(f"Statistics saved to {self.stats_file_path}")
        logger.info(f"Eval saved to {self.eval_file_path}")

    def process(self) -> None:
        self.add_scores()
        if self.extensive:
            self.add_scores_extensive()
        stats = self.calculate_statistics()
        self.save_results(stats)


def batch_process_dir_path(
    input_dir_path: str,
    output_dir_path: Optional[str] = None,
    metrics: Optional[list[str]] = None,
    extensive: bool = False,
) -> None:
    input_path = Path(input_dir_path)
    output_path = Path(output_dir_path) if output_dir_path else input_path
    if not output_path.exists() or not output_path.is_dir():
        print(f"Creating output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    print(f"Processing files in {input_path} and saving results to {output_path}")
    print(f"Metrics to calculate: {metrics}")
    print(f"Files to process: {len(list(input_path.glob('*.json')))}")
    files = list(input_path.glob("*.json"))
    for file in files:
        if file.stem.endswith(("-eval", "-stats")):
            continue

        output_file = output_path / file.name
        logger.info(f"Processing {file}")

        evaluator = Evaluator(str(file), str(output_file), metrics, extensive)
        evaluator.process()

    logger.info("Batch processing completed.")


def main(
    input_json_file_path: str,
    output_dir_path: Optional[str] = None,
    metrics: Optional[list[str]] = None,
    batch: bool = False,
    extensive: bool = False,
) -> None:
    if batch:
        batch_process_dir_path(
            input_json_file_path, output_dir_path, metrics, extensive
        )
    else:
        evaluator = Evaluator(input_json_file_path, output_dir_path, metrics, extensive)
        evaluator.process()


if __name__ == "__main__":
    fire.Fire(main)

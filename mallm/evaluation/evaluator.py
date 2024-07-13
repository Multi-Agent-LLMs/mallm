import json
import logging
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
        extensive: bool = False,
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

        self.extensive = extensive

    def calculate_scores(self, answer: str, references: list[str]) -> dict[str, Any]:
        if references:
            metrics = self.metrics
        elif "answerability" in [metric.name for metric in self.metrics]:
            metrics = [AnswerabilityBoolean()]
        elif "squad" in [metric.name for metric in self.metrics]:
            metrics = [SquadScore()]
        else:
            logger.warning("No metrics to evaluate.")
            return {}

        scores: dict[str, Any] = {}
        for metric in metrics:
            scores = {**scores, **metric.evaluate(answer, references)}
        return scores

    def add_scores(self) -> None:
        for item in tqdm(self.data):
            answer = item.get("answer", "")
            references = item.get("references", [])
            if answer:
                score = self.calculate_scores(answer, references)
                item["scores"] = score

    def add_scores_extensive(self) -> None:
        for item in tqdm(self.data):
            for mem in item.get("globalMemory", []):
                solution = mem.get("solution", "")
                references = item.get("references", [])
                if solution:
                    score = self.calculate_scores(solution, references)
                    mem["scores"] = score

    def calculate_statistics(self) -> None:
        # For each numeric metric, calculate the average and standard deviation
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
            scores = [item.get("scores", {}).get(metric, None) for item in self.data]
            scores = [score for score in scores if score is not None]
            if not scores:
                logger.warning(f"No scores found for {metric}")
                continue

            if not all(isinstance(score, (int, float)) for score in scores):
                logger.info(f"Scores for {metric} are not numeric")
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

        stats_output_file_path = self.output_file_path.replace(".json", "_stats.json")
        with open(stats_output_file_path, "w") as file:
            json.dump(stats, file, indent=4)
        logger.info(f"Statistics saved to {stats_output_file_path}")

    def save_json(self) -> None:
        with open(self.output_file_path, "w") as file:
            json.dump(self.data, file, indent=4)

    def process(self) -> None:
        self.add_scores()
        if self.extensive:
            self.add_scores_extensive()
        self.calculate_statistics()
        self.save_json()
        logger.info(f"Scores saved to {self.output_file_path}")


def main(
    input_file_path: str,
    output_file_path: Optional[str] = None,
    metrics: Optional[list[str]] = None,
    extensive: bool = False,
) -> None:
    evaluator = Evaluator(input_file_path, output_file_path, metrics, extensive)
    evaluator.process()


if __name__ == "__main__":
    fire.Fire(main)

import json
from metrics.bleu import BLEU
from metrics.rouge import ROUGE
from metrics.bertscore import BERTScore
from metrics.meteor import METEOR
import fire
from tqdm import tqdm


class ScoreCalculator:
    def __init__(self, input_file_path: str, metrics: list[str] = ["bleu"]):
        self.input_file_path = input_file_path
        self.data = self.load_json()

        all_metrics = [BLEU(), ROUGE(), BERTScore(), METEOR()]
        metrics = [m.lower() for m in metrics]

        self.metrics = []
        for metric in all_metrics:
            if metric.get_metric_name().lower() in metrics:
                self.metrics.append(metric)

    def load_json(self):
        with open(self.input_file_path, "r") as file:
            return json.load(file)

    def calculate_scores(self, answer, references):
        # Tokenize the answer and references
        scores = {}
        for metric in self.metrics:
            scores = scores | metric.evaluate(answer, references)
        return scores

    def add_scores(self):
        for item in tqdm(self.data):
            answer = item.get("answer", "")
            references = item.get("references", [])
            if answer and references:
                score = self.calculate_scores(answer, references)
                item["scores"] = score

    def save_json(self, output_file_path):
        with open(output_file_path, "w") as file:
            json.dump(self.data, file, indent=4)

    def process(self, output_file_path):
        self.add_scores()
        self.save_json(output_file_path)


def main(input_file_path: str, output_file_path: str, metrics: list[str]) -> None:
    print("Metrics to calculate: " + str(metrics))
    evaluator = ScoreCalculator(input_file_path, metrics)
    evaluator.process(output_file_path)
    print(f"Scores added and saved to {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)

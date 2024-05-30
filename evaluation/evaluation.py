import json
from metrics.bleu import BLEU
from metrics.rouge import ROUGE
import fire


class ScoreCalculator:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.metrics = [BLEU, ROUGE]
        self.data = self.load_json()

    def load_json(self):
        with open(self.json_file_path, "r") as file:
            return json.load(file)

    def calculate_scores(self, answer, references):
        # Tokenize the answer and references
        scores = {}
        for metric in self.metrics:
            scores = scores | metric().evaluate(answer, references)
        return scores

    def add_scores(self):
        for item in self.data:
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


def main() -> None:
    # Example usage:
    json_file_path = "test_out.json"
    output_file_path = "test_out_evaluated.json"

    evaluator = ScoreCalculator(json_file_path)
    evaluator.process(output_file_path)

    print(f"Scores added and saved to {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)

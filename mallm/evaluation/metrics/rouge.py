from typing import Any

from rouge_score import rouge_scorer

from mallm.evaluation.metrics.metric import Metric


class ROUGE(Metric):
    """
    A class to evaluate the ROUGE score for text generation tasks.
    """

    _name = "ROUGE"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:
        scorer = rouge_scorer.RougeScorer(
            rouge_types=["rouge1", "rouge2", "rouge3", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(
            target=reference_texts[0], prediction=generated_text
        )  # rouge only takes one reference
        return {
            "rouge1": {
                "precision": scores["rouge1"].precision,
                "recall": scores["rouge1"].recall,
                "fmeasure": scores["rouge1"].fmeasure,
            },
            "rouge2": {
                "precision": scores["rouge2"].precision,
                "recall": scores["rouge2"].recall,
                "fmeasure": scores["rouge2"].fmeasure,
            },
            "rouge3": {
                "precision": scores["rouge3"].precision,
                "recall": scores["rouge3"].recall,
                "fmeasure": scores["rouge3"].fmeasure,
            },
            "rougeL": {
                "precision": scores["rougeL"].precision,
                "recall": scores["rougeL"].recall,
                "fmeasure": scores["rougeL"].fmeasure,
            },
        }

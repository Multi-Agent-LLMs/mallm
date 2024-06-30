from typing import Any

from bert_score import score as bert_score

from mallm.evaluation.metrics.metric import Metric


class BERTScore(Metric):
    """
    A class to evaluate the BERTScore for text generation tasks.
    This includes precision, recall, and F1 score.
    """

    _name = "BERTScore"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:
        # Calculate BERTScore
        P, R, F1 = bert_score(
            cands=[generated_text],
            refs=[reference_texts[0]],
            lang="en",
            model_type="bert-base-uncased",
            num_layers=9,
        )
        return {
            "bertscore": F1.mean().item()
        }

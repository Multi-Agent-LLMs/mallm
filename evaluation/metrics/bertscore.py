from . import metric
from bert_score import score as bert_score


class BERTScore(metric.Metric):
    """
    A class to evaluate the BERTScore for text generation tasks.
    """

    def evaluate(self, generated_text: str, reference_texts: list[str]) -> dict:
        """
        Evaluate the generated text against a reference text using BERTScore.

        Args:
        generated_text (str): The text generated by the model.
        reference_texts (list[str]): The list of reference texts to compare against.

        Returns:
        dict: The BERTScore including precision, recall, and F1 score.
        """
        # Calculate BERTScore
        P, R, F1 = bert_score(
            [generated_text],
            [reference_texts[0]],
            lang="en",
            model_type="bert-base-uncased",
            num_layers=9,
        )
        scores = {
            "bertscore": {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "fmeasure": F1.mean().item(),
            }
        }
        return scores

    def get_metric_name(self) -> str:
        """
        Return the name of the evaluation metric.

        Returns:
        str: The name of the metric, "BERTScore".
        """
        return "BERTScore"

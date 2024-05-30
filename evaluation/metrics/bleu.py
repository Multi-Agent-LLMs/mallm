from . import metric
from datasets import load_metric
from nltk import word_tokenize

# from nltk.translate.bleu_score import sentence_bleu    # preferred but broken with python 3.12 at commit time


class BLEU(metric.Metric):
    """
    A class to evaluate the BLEU score for text generation tasks.
    """

    def evaluate(self, generated_text: str, reference_texts: list[str]) -> float:
        """
        Evaluate the generated text against a reference text using BLEU score.

        Args:
        generated_text (str): The text generated by the model.
        reference_text (str): The reference text to compare against.

        Returns:
        float: The BLEU score.
        """
        # Tokenize the input texts
        generated_tokens = word_tokenize(generated_text)
        reference_tokens = []
        for r in reference_texts:
            reference_tokens.append(word_tokenize(r))
        # Calculate BLEU score
        score = load_metric("bleu").compute(
            references=[reference_tokens], predictions=[generated_tokens]
        )
        return {"bleu": score["bleu"]}

    def get_metric_name(self) -> str:
        """
        Return the name of the evaluation metric.

        Returns:
        str: The name of the metric, "BLEU".
        """
        return "BLEU"

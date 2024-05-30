from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Abstract class for evaluation metrics in text generation tasks.
    This class provides a template for implementing various evaluation metrics.
    """

    @abstractmethod
    def evaluate(self, generated_text: str, reference_texts: list[str]) -> float:
        """
        Evaluate the generated text against a reference text.

        Args:
        generated_text (str): The text generated by the model.
        reference_text (str): The reference text to compare against.

        Returns:
        float: The evaluation score.
        """
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """
        Return the name of the evaluation metric.

        Returns:
        str: The name of the metric.
        """
        pass

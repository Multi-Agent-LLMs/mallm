from abc import ABC, abstractmethod

from mallm.agents.panelist import Panelist
from mallm.utils.types import Agreement


class DecisionProtocol(ABC):
    """
    Abstract base class for a decision protocol in a multi-agent LLM framework.
    Any concrete decision protocol must implement the make_decision method.
    """

    def __init__(self, panelists: list[Panelist]):
        self.panelists = panelists

    @abstractmethod
    def make_decision(
        self, agreements: list[Agreement], turn: int, task: str, question: str
    ) -> tuple[str, bool]:
        """
        Abstract method to make a decision based on agreements, the current turn number, and the list of panelists.

        Parameters:
        agreements (list[dict[str, any]]): A list of agreement objects from agents.
        turn (int): The current turn number.

        Returns:
        str, bool: str is the result of the conversation and bool describes whether they agreed or not.
        """
        pass

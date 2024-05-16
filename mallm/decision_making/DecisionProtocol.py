from abc import ABC, abstractmethod
from typing import List, Dict

from mallm.agents.panelist import Panelist
from mallm.utils.types.Agreement import Agreement


class DecisionProtocol(ABC):
    """
    Abstract base class for a decision protocol in a multi-agent LLM framework.
    Any concrete decision protocol must implement the make_decision method.
    """

    def __init__(self, panelists: List[Panelist]):
        self.panelists = panelists

    @abstractmethod
    def make_decision(self, agreements: List[Agreement], turn: int) -> tuple[str, bool]:
        """
        Abstract method to make a decision based on agreements, the current turn number, and the list of panelists.

        Parameters:
        agreements (List[Dict[str, any]]): A list of agreement objects from agents.
        turn (int): The current turn number.

        Returns:
        str, bool: The decision result.
        """
        pass

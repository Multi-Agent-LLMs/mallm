from mallm.agents.draftProposer import DraftProposer
from mallm.agents.panelist import Panelist
from mallm.coordinator import Coordinator
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator

coordinator = Coordinator(None, None)
response_generator = FreeTextResponseGenerator(None)


def test_panelist_initialization():
    agent = Panelist(None, None, coordinator, response_generator, None, None)
    assert isinstance(agent, Panelist), "Panelist instance is not created properly."


def test_draft_proposer_initialization():
    agent = DraftProposer(None, None, coordinator, response_generator, None, None)
    assert isinstance(agent, DraftProposer), "DraftProposer instance is not created properly."

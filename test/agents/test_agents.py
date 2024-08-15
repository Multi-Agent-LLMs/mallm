from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.coordinator import Coordinator
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator

coordinator = Coordinator(None, None)
response_generator = FreeTextResponseGenerator(None)


def test_panelist_initialization():
    agent = Panelist(None, None, coordinator, response_generator, None, None, {})
    assert isinstance(agent, Panelist), "Panelist instance is not created properly."


def test_moderator_initialization():
    agent = Moderator(None, None, coordinator, response_generator, None, None, {})
    assert isinstance(agent, Moderator), "Moderator instance is not created properly."

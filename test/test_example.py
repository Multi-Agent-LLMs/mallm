import pytest
from mallm.agents.panelist import Panelist
from mallm.agents.moderator import Moderator


@pytest.fixture()
def test_agent_initialization():
    agent = Panelist()
    assert isinstance(agent, Panelist), "Panelist instance is not created properly."


@pytest.fixture()
def test_agent_initialization():
    agent = Moderator()
    assert isinstance(agent, Moderator), "Panelist instance is not created properly."

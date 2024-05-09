import pytest
from mallm.agents.panelist import *
from mallm.agents.moderator import *
from mallm.discourse_policy.coordinator import *

coordinator = Coordinator(None, None)


def test_agent_initialization():
    agent = Panelist(None, None, coordinator, None, None)
    assert isinstance(agent, Panelist), "Panelist instance is not created properly."


def test_agent_initialization():
    agent = Moderator(None, None, coordinator, None, None)
    assert isinstance(agent, Moderator), "Panelist instance is not created properly."

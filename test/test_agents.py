import pytest
from mallm.agents.panelist import *
from mallm.agents.moderator import *
from mallm.discourse_policy.coordinator import *

coordinator = Coordinator(None, None)


def test_panelist_initialization():
    agent = Panelist(None, None, coordinator, None, None)
    assert isinstance(agent, Panelist), "Panelist instance is not created properly."


def test_moderator_initialization():
    agent = Moderator(None, None, coordinator, None, None)
    assert isinstance(agent, Moderator), "Moderator instance is not created properly."

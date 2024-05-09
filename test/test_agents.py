import pytest
from mallm.agents.panelist import *
from mallm.agents.moderator import *


def test_agent_initialization():
    agent = Panelist()
    assert isinstance(agent, Panelist), "Panelist instance is not created properly."


def test_agent_initialization():
    agent = Moderator()
    assert isinstance(agent, Moderator), "Panelist instance is not created properly."

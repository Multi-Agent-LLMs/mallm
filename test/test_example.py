import pytest
from mallm.agents import Agent


@pytest.fixture()
def test_agent_initialization():
    agent = Agent()
    assert isinstance(agent, Agent), "Agent instance is not created properly."

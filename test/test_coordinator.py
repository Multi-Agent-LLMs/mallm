from unittest.mock import Mock

import pytest

from mallm.coordinator import Coordinator
from mallm.utils.config import Config
from mallm.utils.types import Memory


# Test initialization of Coordinator
def test_coordinator_initialization():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock", memory_bucket_dir="./test/data/"
    )
    assert coordinator.llm == model
    assert coordinator.client == client
    assert coordinator.personas is None
    assert coordinator.panelists == []
    assert coordinator.agents == []
    assert coordinator.use_moderator is False
    assert coordinator.moderator is None
    assert coordinator.decision_protocol is None


# Test initialization of agents with a valid PersonaGenerator
def test_init_agents_with_persona_generator():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock", memory_bucket_dir="./test/data/"
    )
    coordinator.init_agents(
        "task_instruction",
        "input_str",
        use_moderator=False,
        num_agents=3,
        chain_of_thought=False,
        feedback_only=False,
    )
    assert len(coordinator.agents) == 3  # TODO This hardcoded value is not good


# Test initialization of agents with an invalid PersonaGenerator
def test_init_agents_with_wrong_persona_generator():
    model = Mock()
    client = Mock()
    agent_generator = "exp"
    coordinator = Coordinator(
        model, client, agent_generator=agent_generator, memory_bucket_dir="./test/data/"
    )
    with pytest.raises(Exception, match="Invalid persona generator."):
        coordinator.init_agents(
            "task_instruction",
            "input_str",
            use_moderator=False,
            num_agents=3,
            chain_of_thought=False,
            feedback_only=False,
        )


# Test updating global memory
def test_update_global_memory():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock", memory_bucket_dir="./test/data/"
    )
    memory = Memory(
        message_id=1,
        message="content",
        agent_id="agent1",
        agreement=True,
        solution="draft",
        memory_ids=[1],
        additional_args={},
        turn=1,
        persona="test",
        contribution="contribution",
    )
    coordinator.update_global_memory(memory)
    retrieved_memory = coordinator.get_global_memory()
    assert len(retrieved_memory) == 1
    assert retrieved_memory[0].message_id == 1
    assert retrieved_memory[0].message == "content"
    assert retrieved_memory[0].agent_id == "agent1"


def test_update_global_memory_fail():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock", memory_bucket_dir="./test/data_invalid/"
    )
    memory = Memory(
        message_id=1,
        message="content",
        agent_id="agent1",
        agreement=True,
        solution="draft",
        memory_ids=[1],
        additional_args={},
        turn=1,
        persona="test",
        contribution="contribution",
    )
    with pytest.raises(Exception, match="No such file or directory"):
        coordinator.update_global_memory(memory)


# Test updating memories of agents
def test_update_memories():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock", memory_bucket_dir="./test/data/"
    )
    coordinator.init_agents(
        task_instruction="task_instruction",
        input_str="input_str",
        use_moderator=False,
        num_agents=3,
        chain_of_thought=False,
        feedback_only=False,
    )
    memories = [
        Memory(
            message_id=1,
            message="content",
            agent_id="agent1",
            agreement=True,
            solution="draft",
            memory_ids=[1],
            additional_args={},
            turn=1,
            persona="test",
            contribution="contribution",
        )
    ]
    coordinator.update_memories(memories, coordinator.agents)
    for agent in coordinator.agents:
        assert len(agent.get_memories()[0]) == 1
        assert agent.get_memories()[0][0].message_id == 1
        assert agent.get_memories()[0][0].message == "content"
        assert agent.get_memories()[0][0].agent_id == "agent1"


# Test discuss method with invalid paradigm
def test_discuss_with_invalid_paradigm():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock", memory_bucket_dir="./test/data/"
    )
    with pytest.raises(
        Exception, match="No valid discourse policy for paradigm invalid_paradigm"
    ):
        coordinator.discuss(
            Config(
                data="",
                out="",
                instruction="task_instruction",
                paradigm="invalid_paradigm",
                decision_protocol="majority_consensus",
                num_agents=3,
            ),
            ["input_str"],
            [],
        )


# Test discuss method with invalid decision protocol
def test_discuss_with_invalid_decision_protocol():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock", memory_bucket_dir="./test/data/"
    )
    with pytest.raises(
        Exception, match="No valid decision protocol for invalid_protocol"
    ):
        coordinator.discuss(
            Config(
                data="",
                out="",
                instruction="task_instruction",
                paradigm="memory",
                decision_protocol="invalid_protocol",
                num_agents=3,
            ),
            ["input_str"],
            [],
        )

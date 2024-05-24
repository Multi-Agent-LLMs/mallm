from unittest.mock import Mock

import pytest

from mallm.coordinator import Coordinator
from mallm.utils.types import Memory


# Test initialization of Coordinator
def test_coordinator_initialization():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(model, client, memory_bucket_dir="./test/data/")
    assert coordinator.llm == model
    assert coordinator.client == client
    assert coordinator.personas is None
    assert coordinator.panelists == []
    assert coordinator.agents == []
    assert coordinator.use_moderator is False
    assert coordinator.moderator is None
    assert coordinator.decision_making is None


# Test initialization of agents with a valid PersonaGenerator
def test_init_agents_with_valid_persona_generator():
    model = Mock()
    client = Mock()
    agent_generator = Mock()
    agent_generator.generate_personas.return_value = [
        {"role": "role1", "description": "desc1"},
        {"role": "role2", "description": "desc2"},
        {"role": "role3", "description": "desc3"},
    ]
    coordinator = Coordinator(
        model, client, agent_generator=agent_generator, memory_bucket_dir="./test/data/"
    )
    coordinator.init_agents("task_instruction", "input_str", use_moderator=False)
    assert len(coordinator.agents) == 3


# Test initialization of agents without a PersonaGenerator
def test_init_agents_without_persona_generator():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(model, client, memory_bucket_dir="./test/data/")
    with pytest.raises(Exception, match="No persona generator provided."):
        coordinator.init_agents("task_instruction", "input_str", use_moderator=False)


# Test updating global memory
def test_update_global_memory():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(model, client, memory_bucket_dir="./test/data/")
    memory = Memory(
        message_id=1,
        text="content",
        agent_id="agent1",
        agreement=True,
        extracted_draft="draft",
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
    assert retrieved_memory[0].text == "content"
    assert retrieved_memory[0].agent_id == "agent1"


# Test updating memories of agents
def test_update_memories():
    model = Mock()
    client = Mock()
    agent_generator = Mock()
    agent_generator.generate_personas.return_value = [
        {"role": "role1", "description": "desc1"},
        {"role": "role2", "description": "desc2"},
        {"role": "role3", "description": "desc3"},
    ]
    coordinator = Coordinator(
        model, client, agent_generator=agent_generator, memory_bucket_dir="./test/data/"
    )
    coordinator.init_agents("task_instruction", "input_str", use_moderator=False)
    memories = [
        Memory(
            message_id=1,
            text="content",
            agent_id="agent1",
            agreement=True,
            extracted_draft="draft",
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
        assert agent.get_memories()[0][0].text == "content"
        assert agent.get_memories()[0][0].agent_id == "agent1"


# Test discuss method with invalid paradigm
def test_discuss_with_invalid_paradigm():
    model = Mock()
    client = Mock()
    agent_generator = Mock()
    agent_generator.generate_personas.return_value = [
        {"role": "role1", "description": "desc1"},
        {"role": "role2", "description": "desc2"},
        {"role": "role3", "description": "desc3"},
    ]
    coordinator = Coordinator(
        model, client, agent_generator, memory_bucket_dir="./test/data/"
    )
    with pytest.raises(
        Exception, match="No valid discourse policy for paradigm invalid_paradigm"
    ):
        coordinator.discuss(
            "task_instruction",
            "input_str",
            [],
            False,
            (0, 0),
            "invalid_paradigm",
            "majority_consensus",
            10,
            100,
            False,
            False,
            None,
        )


# Test discuss method with invalid decision protocol
def test_discuss_with_invalid_decision_protocol():
    model = Mock()
    client = Mock()
    agent_generator = Mock()
    agent_generator.generate_personas.return_value = [
        {"role": "role1", "description": "desc1"},
        {"role": "role2", "description": "desc2"},
        {"role": "role3", "description": "desc3"},
    ]
    coordinator = Coordinator(
        model, client, agent_generator, memory_bucket_dir="./test/data/"
    )
    with pytest.raises(
        Exception, match="No valid decision protocol for invalid_protocol"
    ):
        coordinator.discuss(
            "task_instruction",
            "input_str",
            [],
            False,
            (0, 0),
            "memory",
            "invalid_protocol",
            10,
            100,
            False,
            False,
            None,
        )

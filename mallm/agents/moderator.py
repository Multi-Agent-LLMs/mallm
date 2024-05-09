from mallm.agents.agent import *


class Moderator(Agent):
    def __init__(
        self,
        llm,
        client,
        coordinator,
        persona="Moderator",
        persona_description="A super-intelligent individual with critical thinking who has a neutral position at all times. He acts as a mediator between other discussion participants.",
    ):
        self.id = str(uuid.uuid4())
        self.short_id = self.id[:4]
        self.persona = persona
        self.persona_description = persona_description
        self.memory_bucket = coordinator.memory_bucket_dir + "agent_{}".format(self.id)
        self.coordinator = coordinator
        self.moderator = self
        self.llm = llm
        self.client = client

        logger.info(
            f'Creating agent {self.short_id} with personality "{self.persona}": "{self.persona_description}"'
        )


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)

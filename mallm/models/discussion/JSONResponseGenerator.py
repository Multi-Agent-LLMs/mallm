import json
import logging
from typing import Optional

from json_repair import repair_json

from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class JSONResponseGenerator(ResponseGenerator):

    _name = "json"

    def __init__(self, llm: Chat):
        self.llm = llm
        self.base_prompt = {
            "role": "system",
            "content": """
You take part in a discussion. Contribute to the discussion according to your role. Your response must be formatted using the provided JSON schema correctly. Do not include extra text.
The response includes the "agreement", the "message", and the "solution".

Example 1:
Task: Give good ingredients for a cooking recipe.
Input: A unique dessert that combines chocolate and fruit.
Context:
When it comes to desserts, combining chocolate and fruit can create a delightful and refreshing treat. Consider the sweetness of the chocolate and the tartness of the fruit to balance the flavors. You can use a variety of fruits such as strawberries, blueberries, or raspberries to pair with dark, milk, or white chocolate.
Your role: Pastry chef
Current Solution: Chocolate-Dipped Strawberry Tart.
JSON Response:
{"agreement": false, "message": "The current solution, Chocolate-Dipped Strawberry Tart, is a classic combination, but it lacks originality. To truly create a unique dessert, consider pairing chocolate with a less common fruit, such as pomegranate or persimmon, to introduce new flavor profiles and textures.", "solution": "Pomegranate and Dark Chocolate Truffle Cake."}

Example 2:
Task: Find a good rhyme for the poem.
Input: The sun sets slow and paints the sky,
A fiery hue that makes me sigh.
Your Role: Poet
Current Solution: The stars come out and twinkle bright,
A night of rest, a peaceful sight.
JSON Response:
{"agreement": true, "message": "The current solution provides a fitting rhyme and captures the serene atmosphere of the night sky. The use of 'bright' and 'sight' creates a smooth and natural rhyme scheme, enhancing the poem's musicality.", "solution": "The stars come out and twinkle bright, A night of rest, a peaceful sight."}

Example 3:
Task: Answer the question by choosing option A, B, C, or D.
Input: What is the world's largest desert?
Context:
A) The Sahara
B) The Gobi
C) The Mojave
D) The Atacama
Your role: Geographer
Current Solution: B) The Gobi
JSON Response:
{"agreement": false, "message": "The current solution, B) The Gobi, is incorrect. As a geographer I know that the Sahara is widely recognized as the world's largest hot desert, covering most of North Africa. It spans across several countries, including Algeria, Chad, Egypt, Libya, Mali, Mauritania, Morocco, Niger, and Tunisia.", "solution": "A) The Sahara"}
        """,
        }

        self.base_prompt_baseline = {
            "role": "system",
            "content": """
Provide a solution to the task. Your response must be formatted using the provided JSON schema correctly. Do not include extra text.
The response includes the "message" and the "solution".

Example 1:
Task: Paraphrase the input text.
Input: It's a task that would challenge even the sharpest of computer geeks: set up a hacker-proof computer network for 190,000 government workers across the country fighting terrorism.
JSON Response:
{"message": "To paraphrase this text, I incorporate spelling changes ('incremental' -> 'further'). I also remove a comma after 'San Francisco'. By adding 'marking' and 'ongoing' the paraphrase appears more clear and detailed.", "solution": "The incremental step, reported by researchers at UC San Francisco, is the latest in a decade-long effort to infect mice with the virus."}

Example 2:
Task: Answer the question.
Input: On which floor is the Belgian Suite located?
Context: When paying a state visit to Britain, foreign heads of state are usually entertained by the Queen at Buckingham Palace. They are allocated a large suite of rooms known as the Belgian Suite, situated at the foot of the Minister's Staircase, on the ground floor of the north-facing Garden Wing. The rooms of the suite are linked by narrow corridors, one of them is given extra height and perspective by saucer domes designed by Nash in the style of Soane. A second corridor in the suite has Gothic influenced cross over vaulting. The Belgian Rooms themselves were decorated in their present style and named after Prince Albert's uncle Léopold I, first King of the Belgians. In 1936, the suite briefly became the private apartments of the palace when they were occupied by King Edward VIII.
JSON Response:
{"message": "The context clearly indicates that the belgian suite is located on the ground floor.", "solution": "The ground floor."}

Example 3:
Task: Answer the question by choosing option A, B, C, or D.
Input: Imagine that there are two influential artificial intelligence (AI) researchers. Let's call them Maria and Anna. They have different opinions on the topic of AI safety. Maria thinks that worrying about AI safety is like worrying about overpopulation on Venus, as we are too far from creating a powerful AI for it to be an issue. Anna thinks the topic of AI safety should already be a top priority, as even today's AIs could harm many humans, if improperly designed. Which opinion is less likely to result in an unforeseen risk?
Context: A) it was not a right thing to do
B) it was a right thing to do
C) the question is not relevant to my goals
D) I prefer not to say
JSON Response:
{"message": "The dying man explicitly stated his wish to be cremated, but his family decided to bury him instead. This goes against his explicit desire. Most humans would likely think that the family's decision is: Disrespectful to the deceased's wishes. They would probably feel that the family should have honored the man's last wish, especially since he took the time to write it down in his testament.", "solution": "A) it was not a right thing to do"}
        """,
        }

    def generate_baseline(
        self, task_instruction: str, input_str: str, chain_of_thought: bool
    ) -> Response:
        prompt_content = f"""
Task: {task_instruction}
Input: {input_str}
        """  # input has context appended
        prompt = [
            self.base_prompt_baseline,
            {
                "role": "user",
                "content": prompt_content,
            },
        ]
        return self.generate_response(prompt, chain_of_thought, None, True, True)

    def generate_response(
        self,
        current_prompt: list[dict[str, str]],
        chain_of_thought: bool,
        agreement: Optional[bool],
        baseline: bool,
        drafting: bool,
    ) -> Response:
        if chain_of_thought:
            current_prompt.append(
                {
                    "role": "assistant",
                    "content": "Let's think step by step.",
                }
            )
        logger.debug(f"Sending prompt: {json.dumps(current_prompt, indent=2)}")

        retry = 0
        while retry < 10:
            try:
                res = self.llm.invoke(current_prompt)
                res = repair_json(res)
                json_response = json.loads(res)
                logger.debug(json_response)
                if isinstance(json_response, list):
                    json_response = json_response[0]

                if (
                    not isinstance(json_response, dict)
                    or "agreement" not in json_response.keys()
                    or "message" not in json_response.keys()
                    or "solution" not in json_response.keys()
                ):
                    retry += 1
                    logger.debug(
                        f"Json missing some keys (will attempt retry no. {retry!s})."
                    )
                    continue

                response = Response(
                    agreement=None if baseline else json_response["agreement"],
                    message=json_response["message"],
                    solution=json_response["solution"],
                )
                if response.agreement is None and not drafting and not baseline:
                    retry += 1
                    continue
                break  # success
            except json.decoder.JSONDecodeError as e:
                retry += 1
                logger.debug(
                    f"Could not decode json (will attempt retry no. {retry!s}): "
                    + str(e)
                    + "\nResponse string: "
                    + str(res)
                )
                continue
        if retry >= 10:
            logger.error(
                f"After 10 retries the json response could not be decoded. \nPrompt: {current_prompt} \nResponse string: {response}"
            )
            raise Exception("Could not decode json.")
        return response

    def generate_feedback(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        current_prompt = [
            self.base_prompt,
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": "Based on the current solution, give constructive feedback.",
            },
        ]
        return self.generate_response(
            current_prompt, chain_of_thought, None, False, False
        )

    def generate_improve(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        current_prompt = [
            self.base_prompt,
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": "Improve the current solution. Agree or disagree and explain your choice.",
            },
        ]
        return self.generate_response(
            current_prompt, chain_of_thought, None, False, False
        )

    def generate_draft(self, data: TemplateFilling, chain_of_thought: bool) -> Response:
        current_prompt = [
            self.base_prompt,
            *self.get_filled_template(data),
            {"role": "user", "content": "Propose a solution to the task."},
        ]
        return self.generate_response(
            current_prompt, chain_of_thought, None, False, True
        )

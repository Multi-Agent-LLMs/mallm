def identify_personas():
    template = '''
    <<SYS>>When faced with a task, begin by identifying the participants who will contribute to solving the task. Provide profiles of the participants, describing their expertise or needs, formatted as a dictionary.

    Here are some examples:
    ---
    Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
    Input: 6 12 1 1

    Profiles: {{
        "Puzzle master": "A super-intelligent person who is capable of solving complex puzzles that require high cognitive skills.",
        "Math expert": "A person who is good at math games, arithmetic calculation, and long-term planning."
    }}
    ---
    Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines, and the first letters of each line form the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten year old kid.

    Profiles: {{
        "Poet": "A person who studies and creates poetry. The poet is familiar with the rules and formats of poetry and can provide guidance on how to write a poem.",
        "Computer Scientist": "A scholar who specializes in the academic study of computer science. The computer scientist is familiar with the concept of a quantum computer and can provide guidance on how to explain it.",
        "Ten year old child": "A child with a limited English vocabulary and little knowledge about complicated concepts, such as a quantum computer."
    }}
    ---

    Now, identify participants relevant to the task and provide their profiles as a dictionary. Remember to present your final output after the prefix "Profiles:". Format the result as a dictionary within curly braces { }.

    Task: {task_instruction} 
    Input: {input}

    Profiles: <</SYS>>'''
    return template
"""
Contains the agent class that we will be using to interact with the agents
"""

class Agent():
    def __init__(self, name="", instruction="", prompt="You are a helpful agent.", model="gpt-4o") -> None:
        self.name = name
        self.instruction = instruction
        self.prompt = prompt
        self.model = model

    def __repr__(self) -> str:
        return f"Agent({self.name})"
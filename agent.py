"""
Contains the agent class that we will be using to interact with the agents
"""
from llama_index.llms.azure_openai import AzureOpenAI

api_key = "895cedddf4a348c3b78f2ad9f7807766"
api_version = "2023-12-01-preview"
azure_endpoint = "https://61a-bot-canada.openai.azure.com"
deployment_name = '61a-bot-prod-gpt4'

class Pipeline:
    def __init__(self, iters, blocks):
        self.iters = iters
        self.blocks = blocks
        self.problem_agent = None
        self.solver_agent = None
        self.verifier_agent = None

    def set_agents(self, problem_agent, solver_agent, verifier_agent):
        self.problem_agent = problem_agent
        self.solver_agent = solver_agent
        self.verifier_agent = verifier_agent

    def run(self, summary):
        print("-------------------------")
        for _ in range(self.iters):
            problem = self.problem_agent.generate_problem(summary)
            print("Generated Problem: ", problem)
            solution = self.solver_agent.solve(problem)
            print("Generated Solution: ", solution)
            is_correct, feedback = self.verifier_agent.verify(problem, solution)
            print(f"correct: {is_correct} feedback: {feedback}")

            if not is_correct:
                # Provide feedback to problem generator
                print(f"Solution was incorrect. Feedback: {feedback}")
                self.problem_agent.update_with_feedback(feedback)

            else:
                print(f"Solution was correct: {solution}")
                break  # Exit early if solution is correct

class Agent:
    def __init__(self, name="", instruction="", prompt="You are a helpful agent.", model_name="gpt-4o"):
        self.name = name
        self.instruction = instruction
        self.prompt = prompt
        self.model_name = model_name
        self.model = AzureOpenAI(
            api_key=api_key,
            deployment_name=deployment_name,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

    def generate_problem(self, summary):
        # Implement logic to generate a problem using the summary
        problem_prompt = f"{self.prompt} Generate a practice problem from the following summary: {summary}"
        problem = self.model.call(problem_prompt)
        return problem

    def solve(self, problem):
        # Implement logic to solve the problem
        solve_prompt = f"{self.prompt} Solve the following problem: {problem}"
        solution = self.model.call(solve_prompt)
        return solution

    def verify(self, problem, solution):
        # Implement logic to verify the solution
        verify_prompt = f"{self.prompt} Verify if the solution is correct for the problem:\nProblem: {problem}\nSolution: {solution}"
        verification = self.model.call(verify_prompt)
        # Parse the verification result (expected to be in format "correct/incorrect")
        is_correct = "correct" in verification.lower()
        feedback = verification if not is_correct else None
        return is_correct, feedback

    def update_with_feedback(self, feedback):
        # Logic to incorporate feedback into the model (fine-tuning prompt or other adjustments)
        print(f"Updating with feedback: {feedback}")
        # Example: adjusting prompt based on feedback
        self.prompt += f" Consider the feedback: {feedback}"

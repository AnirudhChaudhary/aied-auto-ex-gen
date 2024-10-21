# Import the Pipeline and Agent classes
# from agent import Pipeline, Agent

# Set up the Azure OpenAI service
api_key = "895cedddf4a348c3b78f2ad9f7807766"
api_version = "2023-12-01-preview"
azure_endpoint = "https://61a-bot-canada.openai.azure.com"
deployment_name = '61a-bot-prod-gpt4'
import os
import json
from openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI

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

    def run(self, summary, prev_prob):
        print("-------------------------")
        first_problem = True
        for _ in range(self.iters):
            # if first_problem:
            #     problem = self.problem_agent.generate_problem(summary, "")
            #     first_problem = False
            #     print("First Generated Problem: ", problem)
            # else:
            #     problem = self.problem_agent.generate_problem(summary, prev_prob)
            #     print("Generated Problem: ", problem)

            # these should all return strings
            problem = self.problem_agent.generate_problem(summary, prev_prob)
            print("---------------------------------------------------")
            solution = self.solver_agent.solve(problem)
            print("Generated Solution: ", solution)
            print("---------------------------------------------------")
            is_correct, feedback = self.verifier_agent.verify(problem, solution)
            print(f"correct: {is_correct} feedback: {feedback}")
            print("---------------------------------------------------")
            if not is_correct:
                # Provide feedback to problem generator
                print(f"Solution was incorrect. Feedback: {feedback}")
                self.problem_agent.update_with_feedback(feedback)
                prev_prob = problem

            else:
                print(f"Solution was correct: {solution}")
                break  # Exit early if solution is correct
        print("---------------------------------------------------")
        print("final generated problem: ", problem)

class Agent:
    def __init__(self, name="", instruction="", prompt="You are a teacher, teaching a course on Python.", model_name="gpt-4o"):
        self.name = name
        self.instruction = instruction
        self.prompt = prompt  
        self.model_name = model_name
        # self.model = OpenAI(
            #set key here
        # )
    def call(self, message, prompt="", instruction="", tool_choice=False, tools={}):
        """
        Makes the actual call to gpt with the problem prompt and later, if we want tool_choice and tools.
        Input:
        - message : (str) - A message that you want to gpt to act on 
        - prompt : (str) - System defined prompt for gpt. This will be used as context for gpt.
        - instruction : (str) - This is the "prompt" in a traditional setting, it gives local context to your question / statement
        - tool_choice : (bool) - Determines whether or not you want gpt to consider function calls
        - tools : (Dict) - also known as `helper functions` that the LLM can use to answer the prompt

        Output:
        - ChatCompletion() [ Essentially a dictionary ]
        """
        assert type(message) == str, "message should be a string, it is not"
        if prompt == "":
            prompt = self.prompt

        print("-------- CALLING GPT with the following params ----------")
        print("[SYSTEM]: ", prompt)
        print("[USER]: ", instruction + message)

        message_to_send =[
            {"role": "system", "content": prompt},
            {"role": "user", "content": instruction + message}
        ]
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_to_send
        )

        self.log_meta_data_info(response)
        return response

    def log_meta_data_info(self, ChatCompletionResponse):
        """
        Keeps metadata information about the runs that are happening.
        """
        # Define the metadata folder path
        meta_folder = "metadata"

        # Create the metadata folder if it doesn't exist
        if not os.path.exists(meta_folder):
            os.makedirs(meta_folder)

        # Define the metadata file path
        meta_file = os.path.join(meta_folder, "meta.json")

        # Read the metadata file if it exists
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
        else:
            # otherwise just instantiate the info, and this will be added to later on
            meta_data = {}

        # Get the current agent's name
        agent_name = self.name

        # Get the completion tokens, prompt tokens, and total tokens from the ChatCompletionResponse
        completion_tokens = ChatCompletionResponse.usage.completion_tokens
        prompt_tokens = ChatCompletionResponse.usage.prompt_tokens
        total_tokens = ChatCompletionResponse.usage.total_tokens

        # Update the metadata for the current agent

        # instantiate if it is not in teh the meta information yet
        if agent_name not in meta_data:
            meta_data[agent_name] = {}
            meta_data[agent_name]["completion_tokens"] = 0
            meta_data[agent_name]["prompt_tokens"] = 0
            meta_data[agent_name]["total_tokens"] = 0
            meta_data[agent_name]["invocations"] = 0        # this is the number of times we have called this agent

        meta_data[agent_name]["completion_tokens"] += completion_tokens
        meta_data[agent_name]["prompt_tokens"] += prompt_tokens
        meta_data[agent_name]["total_tokens"] += total_tokens
        meta_data[agent_name]["invocations"] += 1

        # Write the updated metadata back to the file
        with open(meta_file, "w") as f:
            json.dump(meta_data, f)
        
        return

    def parse_output(self, response, content=True, function_call=False):
        """
        Parses the output
        """
        if type(response) == str:
            print("response is a string? : ", response)
            return response
        return response.choices[0].message.content


    def generate_problem(self, summary, prev_prob):
        # Implement logic to generate a problem using the summary
        # problem_prompt = f"{self.prompt} Generate a practice problem from the following summary: {summary}"
        
        print("-------- GENERATING PROBLEM ------------")
        base_prompt = f"Generate a practice problem from the following summary: {summary}. The problem should be standalone, allowing me to directly input it into a test. "
        if prev_prob:
            prompt = base_prompt + f"Attached are also some examples of problems that have been used in this section. Please consider the structure of these problems, but do not copy them exactly! {prev_prob}"
            message = ""    # little jank, but everything is sent as part of the prompt
        else:
            prompt=base_prompt
            message = summary
        problem = self.call(message, instruction=prompt)
        problem = self.parse_output(problem)
        return problem

    def solve(self, problem):
        # Implement logic to solve the problem
        # solve_prompt = f"{self.prompt} Solve the following problem: {problem}"
        print("-------- SOLVING PROBLEM ------------")
        prompt = "You are an expert solver. You look at the questions, think about the correct solution, and return only the solution to the questions without the explanations."
        instruction = "Answer the following questions: "
        message = problem
        solution = self.call(message=message, prompt=prompt, instruction=instruction)
        solution = self.parse_output(solution)
        return solution

    def verify(self, problem, solution):
        # Implement logic to verify the solution
        # verify_prompt = f"{self.prompt} Verify if the solution is correct for the problem:\nProblem: {problem}\nSolution: {solution}"
        print("-------- VERIFYING PROBLEM ------------")
        problem_solution_message = f"\nProblem: {problem}\nSolution: {solution}"
        prompt = "You are an expert verifier. You look at the questions and check whether or not the solution is correct."
        instruction = "Verify that the solutions answer the problem. "
        message = problem_solution_message
        verification = self.call(message=message, prompt=prompt, instruction=instruction)
        verification = self.parse_output(verification)
        # Parse the verification result (expected to be in format "correct/incorrect")
        is_correct = "correct" in verification.lower()
        feedback = verification if not is_correct else None
        return is_correct, feedback

    def update_with_feedback(self, feedback):
        # Logic to incorporate feedback into the model (fine-tuning prompt or other adjustments)
        print(f"Updating with feedback: {feedback}")
        # Example: adjusting prompt based on feedback
        self.prompt += f" Consider the feedback: {feedback}"

# Create the problem generator, solver, and verifier agents
problem_agent = Agent(name="Problem Generator", instruction="Generate a practice problem from the following summary.", model_name="gpt-4o")
print("Problem Generator Agent Created")
solver_agent = Agent(name="Solver", instruction="Solve the following problem.", model_name="gpt-4o")
print("Solver Agent Created")
verifier_agent = Agent(name="Verifier", instruction="Verify if the solution is correct for the problem.", model_name="gpt-4o")
print("Verifier Agent Created")

# Create the pipeline and set the agents
pipeline = Pipeline(iters=2, blocks=[])
pipeline.set_agents(problem_agent, solver_agent, verifier_agent)

# Define the summary of the chapters
summary = """Higher-order functions: Passing functions as arguments enhances expressiveness but can clutter the global frame with many unique function names and requires specific argument signatures.

Nested function definitions: Defining functions within other functions addresses issues with name collisions and argument compatibility, as these local functions are only accessible during the enclosing function's execution. Local definitions are evaluated only when the enclosing function is called.

Lexical scoping:

Inner functions can access names defined in their enclosing scope.
Functions share the enclosing environment's names rather than the calling environment’s.
Environment model extensions:

User-defined functions now include a "parent environment" annotation, indicating where they were defined.
Nested function calls create extended environments with chains of frames that can be arbitrarily long, ending at the global frame.
Frame resolution: When evaluating a nested function, Python resolves variable names by searching through the chain of frames, starting from the current frame and moving up to the parent frames.

Advantages of lexical scoping:

Local function names do not interfere with external names.
Inner functions can access variables from the enclosing environment.
Closures: Functions defined within other functions retain access to the data from their defining environment, effectively "enclosing" this data."""

previous_problems = """def curried_pow(x):
        def h(y):
            return pow(x, y)
        return h
>>> curried_pow(2)(3)"""

# Run the pipeline
print("running the pipeline")

pipeline.run(summary, previous_problems)
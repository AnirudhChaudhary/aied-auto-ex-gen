# Import the Pipeline and Agent classes
# from agent import Pipeline, Agent

# Set up the Azure OpenAI service
import os
import json
from openai import OpenAI

class Pipeline:
    def __init__(self, iters, blocks):
        self.iters = iters
        self.blocks = blocks
        self.problem_agent = None
        self.solver_agent = None
        self.verifier_agent = None

    def set_agents(self, problem_agent, solver_agent, verifier_agent, breaker_agent, comprehendor_agent, eval_agent):
        self.problem_agent = problem_agent
        self.solver_agent = solver_agent
        self.verifier_agent = verifier_agent
        self.breaker_agent = breaker_agent
        self.comprehendor_agent = comprehendor_agent
        self.eval_agent = eval_agent    

    def run(self,prev_prob):
        print(f"------------------------- RUNNING PIPELINE FOR {self.iters} ITERATIONS -----------------------------")
        for _ in range(self.iters):
            # these should all return strings
            print("-------------------------GENERATING PROBLEM---------------------------")
            instruction="Be concise and specific. What concepts is this problem trying to test?"
            raw_question_concepts = self.comprehendor_agent.call(message=prev_prob, system_instruction=instruction)
            question_concepts = Agent.parse_output(raw_question_concepts)

            print("Question Concepts: ", question_concepts)

            difficulty = "same"
            instruction="You are a computer science professor that is trying to create a new midterm problem. There are multiple ways to change a problem, including changing variable names, changing function names, changing the constants, reversing the polarity of the question, or changing a data type. "
            prompt=f"Generate and return another problem of {difficulty} difficulty as the following problem without any greetings: "
            raw_problem = self.breaker_agent.call(message=prev_prob, system_instruction=instruction, llm_prompt=prompt)
            problem = Agent.parse_output(raw_problem)
            print("Tweaked Problem: ", problem)

            instruction="You are a question evaluator. You will be given the concepts the question should test and a question. You will analyze the concepts and you will evaluate if the question still tests the concepts. Return yes or no. If no, explain what is missing from the question."
            prompt = f"Concepts: {question_concepts}\nQuestion: {problem}"
            feedback = self.eval_agent.call(message=prev_prob, system_instruction=instruction, llm_prompt=prompt)
            feedback = Agent.parse_output(feedback)
            print("Feedback: ", feedback)

            print("-------------------------SOLVING PROBLEM---------------------------")
            instruction = "You are an expert solver. You look at the questions, think about the correct solution, and return only the solution to the questions without the explanations."
            prompt = "Answer the following question: "
            solution = self.solver_agent.call(message=problem, system_instruction=instruction, llm_prompt=prompt)
            print("Generated Solution: ", solution)

            print("-------- VERIFYING PROBLEM ------------")
            prompt = "You are an expert verifier. You look at the questions and check whether or not the solution is correct."
            instruction = "Verify that the solutions answer the problem. "
            problem_solution_message = f"\nProblem: {problem}\nSolution: {solution}"

            verification = self.verifier_agent.call(message=problem_solution_message, system_instruction=instruction, llm_prompt=prompt)
            verification = Agent.parse_output(verification)
            # Parse the verification result (expected to be in format "correct/incorrect")
            is_correct = "correct" in verification.lower()
            feedback = verification if not is_correct else None
            print(f"correct: {is_correct} feedback: {feedback}")
            print("---------------------------------------------------")
            if not is_correct:
                # Provide feedback to problem generator
                print(f"Solution was incorrect. Feedback: {feedback}")
                # self.problem_agent.update_with_feedback(feedback)
                # prev_prob = problem

            else:
                print(f"Solution was correct: {solution}")
                break  # Exit early if solution is correct
        # print("---------------------------------------------------")
        # print("final generated problem: ", problem)
        return problem

class Agent:
    def __init__(self, name="", sys_instruction="", llm_prompt="You are a teacher, teaching a course on Python.", model_name="gpt-4o"):
        self.name = name
        self.sys_instruction = sys_instruction
        self.prompt = llm_prompt  
        self.model_name = model_name
        self.model = OpenAI()

    def call(self, message, system_instruction="", llm_prompt="", tool_choice=False, tools={}):
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
        assert type(message) == str, f"message should be a string, it is of type {type(message)}"
        if system_instruction == "":
            system_instruction = self.sys_instruction

        print("-------- CALLING GPT ----------")
        print("[SYSTEM]: ", system_instruction)
        print("[USER]: ", llm_prompt + message)

        message_to_send =[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": llm_prompt + message}
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

    def parse_output(response, content=True, function_call=False):
        """
        Parses the output
        """
        if type(response) == str:
            print("response is a string? : ", response)
            return response
        
        # print("type of response: ", type(response))
        if "choices" not in response:
            print("response does not have choices. It looks like this: ", response)
        return response.choices[0].message.content

# Create the problem generator, solver, and verifier agents
problem_agent = Agent(name="Problem Generator", sys_instruction="Generate a practice problem from the following summary.", model_name="gpt-4o")
solver_agent = Agent(name="Solver", sys_instruction="Solve the following problem.", model_name="gpt-4o")
verifier_agent = Agent(name="Verifier", sys_instruction="Verify if the solution is correct for the problem.", model_name="gpt-4o")
comprehendor_agent = Agent(name="Comprehendor", model_name="gpt-4o")
breaker_agent = Agent(name="Breaker", model_name="gpt-4o")
eval_agent = Agent(name="Question Evaluator", model_name="gpt-4o")

# Create the pipeline and set the agents
pipeline = Pipeline(iters=1, blocks=[])
pipeline.set_agents(problem_agent, solver_agent, verifier_agent, comprehendor_agent, breaker_agent, eval_agent)

# Define the summary of the chapters

previous_problems = """def curried_pow(x):
        def h(y):
            return pow(x, y)
        return h
>>> curried_pow(2)(3)"""

# Run the pipeline
print("running the pipeline")

new_problem = pipeline.run(previous_problems)

print("----------- NEW GENERATED PROBLEM --------------")
print(new_problem)
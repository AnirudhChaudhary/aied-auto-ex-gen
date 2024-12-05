# Import the Pipeline and Agent classes
# from agent import Pipeline, Agent

# Set up the Azure OpenAI service
import os
import json
from openai import OpenAI
import subprocess
import re
import tempfile  # Import tempfile for temporary file handling
import doctest 
import sys
import io
import ast

class Pipeline:
    def __init__(self, iters, blocks):
        self.iters = iters
        self.blocks = blocks
        self.problem_agent = None
        self.solver_agent = None
        self.verifier_agent = None

    def set_agents(self, problem_agent, solver_agent, verifier_agent, qg_agent, comprehendor_agent, eval_agent):
        self.problem_agent = problem_agent
        self.solver_agent = solver_agent
        self.verifier_agent = verifier_agent
        self.question_generator_agent = qg_agent
        self.comprehendor_agent = comprehendor_agent
        self.eval_agent = eval_agent   


    def extract_question_concepts(self, prev_prob):
        instruction="Be concise. What, up to two (1-2 word) concepts is this problem trying to test? "
        raw_question_concepts = self.comprehendor_agent.call(message=prev_prob, system_instruction=instruction)
        question_concepts = Agent.parse_output(raw_question_concepts)
        print("Question Concepts: ", question_concepts)
        return question_concepts

    def generate_problems(self, prev_prob):
        num_problems = 3
        difficulty = "same"
        instruction="You are a computer science professor that is trying to create a new midterm problems. There are multiple ways to change a problem, including changing variable names, changing function names, changing the constants, reversing the polarity of the question, or changing a data type. Make sure that the problems have corresponding docstring tests for correctness."
        prompt=f"Generate and return {num_problems} problems of {difficulty} difficulty as the following problem without any greetings, seperated by --- NEW PROBLEM ---: "
        raw_problem = self.question_generator_agent.call(message=prev_prob, system_instruction=instruction, llm_prompt=prompt, temp=0.3)
        problem = Agent.parse_output(raw_problem)
        problem_list = problem.split("--- NEW PROBLEM ---")
        return problem_list
    
    def eval_problem(self, problem, question_concepts):
        instruction="You are a question evaluator. You will be given the concepts the question should test and a question. You will analyze the concepts and you will evaluate if the question still tests the concepts. Return yes or no. If no, only explain what is missing from the question."
        prompt = f"Concepts: {question_concepts}\nQuestion: {problem}"
        feedback = self.eval_agent.call(message="", system_instruction=instruction, llm_prompt=prompt, temp=0.0)
        feedback = Agent.parse_output(feedback)
        valid_problem = "yes" in feedback.lower()     # if we have a valid problem we don't have to go through and tweak the problem
        # print("Feedback: ", feedback)
        return valid_problem, feedback
    
    def fix_prob_with_feedback(self, problem, feedback):
        # if "failed" not in feedback.lower():
        instruction="You are a computer science professor creating a midterm problem but you've received some feedback on your generated problem. Please fix the problem formulation and return the fixed problem, without any greetings or telling me what you fixed. Make sure that you are not answering the question. You may be given a solution, please ignore and only return the reformulated question."
        prompt=f"Fix the following problem: {problem}."
        message=f"The following is the feedback: {feedback}"
        # else:
        #     instruction="You are a computer science professor creating a midterm problem but you've received some feedback on your generated problem. It looks like the generated tests are incorrect! Think carefully about the question and what the output of the problem should be and fix the docstring tests for the failed tests. "
        #     prompt=f"Fix the following problem but keep the original question prompt: {problem}."
        #     message=f"The following is the feedback: {feedback}"
        problem = self.question_generator_agent.call(message=message, system_instruction=instruction, llm_prompt=prompt)
        problem = Agent.parse_output(problem)
        return problem
    
    def docstring_fixer(self, problem, feedback):
        instruction = "You are a computer science professor teaching Python. You notice one of the practice problems you made has incorrect docstring tests. Walk through each test one by one and check if the docstring test is correct by explaining your reasoning. If not, fix the docstring test. If the docstring test is correct, return the fixed problem. If all the docstring tests are correct, return the problem as is. At the end return the fixed result."
        prompt=f"Fix the docstring tests in the following problem. The docstrings tests are denoted by >>>. The following is the feedback: {feedback}."
        message=f"The following is the original problem with the faulty docstrings tests: {problem}."
        problem = self.question_generator_agent.call(message=message, system_instruction=instruction, llm_prompt=prompt)
        problem = Agent.parse_output(problem)
        return problem
    
    def solve_problem(self, problem):
        print("-------------------------SOLVING PROBLEM---------------------------")
        instruction = "You are an expert solver. You look at the questions, think about the correct solution, and add the solution to the question but don't provide the explanations."
        prompt = "Fill in the solution and make sure to keep the original problem and docstring tests content as well. "
        solution = self.solver_agent.call(message=problem, system_instruction=instruction, llm_prompt=prompt)
        solution = Agent.parse_output(solution)
        print("Generated Solution: ", solution)
        return solution
        


    def run(self,prev_prob):
        count = 1
        output_file = f'final_output_{count}.txt'
        while os.path.exists(output_file):
            count += 1
            output_file = f'final_output_{count}.txt'

        print("----------- NEW GENERATED PROBLEM --------------")
        with open(output_file, "a") as f:
            question_concepts = self.extract_question_concepts(prev_prob)
            problem_list = self.generate_problems(prev_prob)

            final_problem_list = []
            print("problem list len: ", len(problem_list))
            # print("Tweaked Problem: ", problem)
            for problem in problem_list[1:]:
                feedback = None
                valid_problem = False
                for i in range(self.iters):
                    # we have generated the problem now we want to evaluate it
                    valid_problem, feedback = self.eval_problem(problem, question_concepts)
                    if valid_problem:
                        break
                    if feedback:
                        problem = self.fix_prob_with_feedback(problem, feedback)
                        print("Tweaked Problem: ", problem)
                        
                
                final_problem_list.append(problem)
                print("Generated Problem: ", problem)

            prob_w_sol_list = []
            for problem in final_problem_list:
                solution = self.solve_problem(problem)
                prob_w_sol_list.append(solution)
            print("Solutions: ", prob_w_sol_list)
            
            print("-------------------------")
            ######### ANEESH YOU CAN START HERE ############
            print("-------- VERIFYING PROBLEM ------------")
            verified_problems = []
            for prob in prob_w_sol_list:
                for fix_iter in range(3):
                    # print("Problem: ", prob)
                
                    pass_test, feedback, failed_testcases = self.test_docstring_tests(prob)
                    if not pass_test:
                        print("Failed Test Cases: ", failed_testcases)
                        prob = self.docstring_fixer(prob, failed_testcases)
                        print("Tweaked Problem: ", prob)
                        solution = self.solve_problem(prob)
                        # pass_test, feedback, failed_testcases = self.test_docstring_tests(solution)
                        # if not pass_test:
                        #     print("FAILED AGAIN, SECOND TIME IS NOT THE CHARM", failed_testcases)
                    else:
                        verified_problems.append(prob)
                    
                problem_lines = prob.split("\\n")
                f.write(f"-----Problem-----:\n")
                for line in problem_lines:
                    f.write(f"{line}\n")
                f.write(f"Passed Tests: {feedback}\n")
                f.write("-------------------------\n")



        return prob_w_sol_list


    def extract_test_cases(self, docstring):
        test_cases = re.findall(r'>>>.*', docstring)
        return test_cases

    def clean_solution_code(self, solution_code):
        code_fence_pattern = r'^```(?:python)?\n([\s\S]*?)\n```"?$'
        match = re.match(code_fence_pattern, solution_code.strip(), re.MULTILINE)
        if match:
            solution_code = match.group(1)
        else:
            solution_code = solution_code.strip('`"\'')
            if solution_code.startswith('python'):
                solution_code = solution_code[len('python'):].lstrip()
        solution_code = solution_code.strip('```')
        solution_code = solution_code.encode('utf-8').decode('unicode_escape')

        return solution_code.strip()

    def test_docstring_tests(self, solution_code):
        solution_code = self.clean_solution_code(solution_code)
        print("Solution code :\n", solution_code)
        print("------END OF SOL CODE-------")
        if ">>>" not in solution_code:
            return False, "No doctests found ", "No doctests found"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".py") as temp_file:
            temp_file.write(solution_code)
            temp_file_path = temp_file.name

        temp_dir = os.path.dirname(temp_file_path)
        sys.path.insert(0, temp_dir)
        module_name = os.path.splitext(os.path.basename(temp_file_path))[0]
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            verbose_output = io.StringIO()
            sys.stdout = verbose_output  # Redirect stdout
            result = doctest.testmod(module, verbose=True)
            sys.stdout = sys.__stdout__  # Restore stdout

            detailed_output = verbose_output.getvalue()  # Get the captured output
            verbose_output.close()

            print("----DETAILED OUTPUT----")
            print(detailed_output)
            print("-----------------------")

            if result.failed == 0:
                return True, f"All {result.attempted} doctests passed", ""
            else:
                return False, f"{result.failed} out of {result.attempted} doctests failed", detailed_output
        except Exception as e:
            print(f"Error while running doctests: {str(e)}")
            return False, f"Error while running doctests: {str(e)}", "Error while running doctests: " + str(e)

class Agent:
    def __init__(self, name="", sys_instruction="", llm_prompt="You are a teacher, teaching a course on Python.", model_name="gpt-4o"):
        self.name = name
        self.sys_instruction = sys_instruction
        self.prompt = llm_prompt  
        self.model_name = model_name
        self.model = OpenAI()

    def call(self, message, system_instruction="", llm_prompt="", temp=0.0, tool_choice=False, tools={}):
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
            messages=message_to_send,
            temperature=temp
        )

        # print("response looks like this now: ", response)

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
        try:
            content = response["choices"][0]["message"]["content"]
        except:
            response = str(response)
            # print("response: ", response)
            content_start = response.index('content=')                   # get the start of the content
            response_first_half_stripped = response[content_start+9:]       # remove everything up until 'content'
            ending_quote_index = response_first_half_stripped.index("refusal=")  # this is the ending quote, but need to be careful that the index is relative to the content
            content = response_first_half_stripped[:ending_quote_index-2]    # 9 is the len(content=') and ending quote index comes from the previous part, with the new relative section

        print("original content: ", content)
        content = Pipeline.clean_solution_code("",content)
        print("content after stripping ```: ", content)

        return content

# Create the problem generator, solver, and verifier agents
problem_agent = Agent(name="Problem Generator", sys_instruction="Generate a practice problem from the following summary.", model_name="gpt-4o")
solver_agent = Agent(name="Solver", sys_instruction="Solve the following problem.", model_name="gpt-4o")
verifier_agent = Agent(name="Verifier", sys_instruction="Verify if the solution is correct for the problem.", model_name="gpt-4o")
comprehendor_agent = Agent(name="Comprehendor", model_name="gpt-4o")
ques_gen_agent = Agent(name="Breaker", model_name="gpt-4o")
eval_agent = Agent(name="Question Evaluator", model_name="gpt-4o")


# Create the pipeline and set the agents
pipeline = Pipeline(iters=2, blocks=[])
pipeline.set_agents(problem_agent, solver_agent, verifier_agent, comprehendor_agent, ques_gen_agent, eval_agent)

# Define the summary of the chapters

previous_problems = """
def strips(t):
'''Yield the paths from the root to a leaf of Tree t that form strips. A strip is a list of integers in which each integer is one more than the last.
    >>> list(strips(Tree(1, [Tree(3, [Tree(4)]), Tree(2, [Tree(2), Tree(3)])])))
    [[1, 2, 3]]
    >>> list(strips(Tree(1, [Tree(2), Tree(2, [Tree(2), Tree(3, [Tree(4)]), Tree(3)])])))
    [[1, 2], [1, 2, 3, 4], [1, 2, 3]]
    '''
    if t.is_leaf():
        yield _______
    for b in t.branches:
        if _______:
            for s in _______:
                yield _______
"""


new_problem_list = pipeline.run(previous_problems)


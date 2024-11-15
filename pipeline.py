# Import the Pipeline and Agent classes
# from agent import Pipeline, Agent

# Set up the Azure OpenAI service
import os
import json
from openai import OpenAI
import subprocess

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

    def run(self,prev_prob):
        print(f"------------------------- RUNNING PIPELINE FOR {self.iters} ITERATIONS -----------------------------")
        print("-------------------------GENERATING PROBLEM---------------------------")
        instruction="Be concise and specific. Do not answer the provided problem. What concepts is this problem trying to test? "
        raw_question_concepts = self.comprehendor_agent.call(message=prev_prob, system_instruction=instruction)
        question_concepts = Agent.parse_output(raw_question_concepts)
        print("Question Concepts: ", question_concepts)
        feedback = None
        valid_problem = False
        for _ in range(self.iters):
            if valid_problem:
                break
            # these should all return strings
            if not feedback:
                difficulty = "same"
                instruction="You are a computer science professor that is trying to create a new midterm problem. There are multiple ways to change a problem, including changing variable names, changing function names, changing the constants, reversing the polarity of the question, or changing a data type. "
                prompt=f"Generate and return another problem of {difficulty} difficulty as the following problem without any greetings: "
                raw_problem = self.question_generator_agent.call(message=prev_prob, system_instruction=instruction, llm_prompt=prompt)
                problem = Agent.parse_output(raw_problem)
                print("Tweaked Problem: ", problem)
            else:
                instruction="You are a computer science professor creating a midterm problem but you've found some bugs. Please fix the problem and return the fixed problem, without any greetings or telling me what you fixed."
                prompt=f"Fix the following problem: {problem}."
                message=f"The following is the feedback: {feedback}"
                problem = self.question_generator_agent.call(message=message, system_instruction=instruction, llm_prompt=prompt)
                problem = Agent.parse_output(problem)

            instruction="You are a question evaluator. You will be given the concepts the question should test and a question. You will analyze the concepts and you will evaluate if the question still tests the concepts. Return yes or no. If no, explain what is missing from the question."
            prompt = f"Concepts: {question_concepts}\nQuestion: {problem}"
            feedback = self.eval_agent.call(message=prev_prob, system_instruction=instruction, llm_prompt=prompt)
            feedback = Agent.parse_output(feedback)
            valid_problem = "yes" in feedback.lower()     # if we have a valid problem we don't have to go through and tweak the problem
            print("Feedback: ", feedback)

        print("-------------------------SOLVING PROBLEM---------------------------")
        instruction = "You are an expert solver. You look at the questions, think about the correct solution, and return only the solution to the questions without the explanations."
        prompt = "Answer the following question in a .py text format taking special care of tabs: "
        solution = self.solver_agent.call(message=problem, system_instruction=instruction, llm_prompt=prompt)
        solution = Agent.parse_output(solution)
        print("Generated Solution: ", solution)

        print("-------- VERIFYING PROBLEM ------------")
        print("----- PARSING THE TEST CASE EXAMPLES -----")

        comment_beginning = problem.index(">>>")                        # this is the start of the test cases
        stripped_beginning = problem[comment_beginning:]                # remove everything until the start of the test cases
        if "'''" in stripped_beginning:                                 # this is the "end" of the test case section 
            comment_end = stripped_beginning.index("'''")       
            ex_test_case = stripped_beginning[:comment_end]
            test_case_lines = ex_test_case.split("\\n")
            print("test case ''': ", test_case_lines)
        elif "\'\'\'" in stripped_beginning:
            comment_end = stripped_beginning.index("\'\'\'")
            ex_test_case = stripped_beginning[:comment_end]
            test_case_lines = ex_test_case.split("\\n")
            print("test case w slash: ", test_case_lines)
        else:
            print("WE GOT A WEIRD STRUCTURE BACK FROM THE PROBLEM GENERATOR AND IT LOOKS LIKE THIS: ", problem)

        # instruction = f"You are an expert verifier. You will be given an incomplete problem and you will generate a few test cases that test the functionality of the program. An example is: {ex_test_case}. Generate your test cases without the >>>"
        # prompt = "Generate these assertion test cases in a text format for the following problem, separated by a newline character. Do not answer the provided problem. "
        # problem_solution_message = f"\nProblem: {problem}\n"

        # test_cases_chat_message = self.verifier_agent.call(message=problem_solution_message, system_instruction=instruction, llm_prompt=prompt)
        # test_cases = Agent.parse_output(test_cases_chat_message)


        # --------- TEST CASE GENERATION ---------
        # ( these are the lines that will go into the file )

        final_lines = []
        num_lines = len(test_case_lines)        # tells us how many lines there are
        i = 0
        while i < num_lines-1:  # we don't want anything past the newline character of the last line
            if test_case_lines[i].lstrip().startswith(">>>"):
                # Then strip ">>> " specifically from the start
                test_case_lines[i] = test_case_lines[i].lstrip()[4:]
                # print(f"test case line {i}: ", test_case_lines[i])
                if ">>>" in test_case_lines[i+1]:   # if we have a >>> in the previous line then we are guaranteed to have a new line...afaik
                    final_lines.append(test_case_lines[i])
                    
                else:
                    final_lines.append("assert " + test_case_lines[i] + " == ")

            else:       # output that we expect
                final_lines[-1] = final_lines[-1] + str(test_case_lines[i].strip(" "))  # we don't want any spaces

            i += 1
            # print(final_lines)
        
        print("----- FINAL ------")
        for line in final_lines:
            print(line)
        
        print("solution: ", solution)
        # print("final lines: ", final_lines)
        is_correct = self.verifier(solution, final_lines)
        print("is correct: ", is_correct)
        print("final problem: ", problem)
        return problem

    def parse_solution_and_write_to_file(self, llm_output, file, ext=".txt"):
        llm_output_beginning_removed = llm_output.strip("```").replace("python", "", 1).strip()
        llm_output_beginning_removed_split = llm_output_beginning_removed.split("\n")
        print("llm split: ", llm_output_beginning_removed_split)

        with open(file,"w") as f:
            for line in llm_output_beginning_removed_split:
                f.write(line)
                f.write("\n")

    def verifier(self, solution, test_cases):
        """
        This function is verifies that the solution passes the provided test cases by running a python intepreter on the code.
        Currently we are working in a python setting where solution is a python function.

        Input:
        - solution : this is the generated `correct` python function
        - test_cases : this is a string of test cases that are separated by newlines

        Output:
        - is_correct : this is a boolean that indicates if the solution is correct
        """
        
        # create a new file and populate it with the solution and test case
        llm_output_beginning_removed = solution.strip("```").replace("python", "", 1).strip()
        llm_output_beginning_removed_split = llm_output_beginning_removed.split("\\n")
        print("llm output after splitting by newline character: ", llm_output_beginning_removed_split)
        # llm_output_beginning_removed_split = [x.strip("\\n") for x in llm_output_beginning_removed_split] # don't think we want anything after the last \n
        print("llm split: ", llm_output_beginning_removed_split)
        llm_output_beginning_removed_split = llm_output_beginning_removed_split[:-1]
        test_file = "test.txt"
        with open(test_file,"w") as f:
            for line in llm_output_beginning_removed_split:
                line = line.replace("\\n","\n")
                f.write(line)
                f.write("\n")
        
            for line in test_cases:
                f.write(line)
                f.write("\n")
            

        # save the file to disk
        

        # run the file using python interpreter
        result = subprocess.run(['python3', test_file], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')

        if "AssertionError" in output:
            output = False
        elif output == "":
            output = True
        else:
            output = False

        # return the result
        return output

# Pipeline.parse_solution_and_write_to_file("```python\ndef cumulative_sum(s):\n    '''Yield the cumulative sum of values from iterator s.'''\n    total = 0\n    for value in s:\n        total += value\n        yield total\n```", )

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
            messages=message_to_send,
            temperature=0.3
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
Implement differences, a generator function that takes t, a non-empty iterator over numbers. It yields the differences between each pair of adjacent values from t. If t iterates over a positive finite number of values n, then differences should yield n-1 times.\n 
def differences(t):
    '''Yield the differences between adjacent values from iterator t.

    >>> list(differences(iter([5, 2, -100, 103])))
    [-3, -102, 203]
    >>> next(differences(iter([39, 100])))
    61
    '''
    "*** YOUR CODE HERE ***"

"""

# Run the pipeline
output_file = "output.txt"

def parse_solution_and_write_to_file(llm_output, file, ext=".txt"):
    llm_output_beginning_removed = llm_output.strip("```").replace("python", "", 1).strip()
    llm_output_beginning_removed_split = llm_output_beginning_removed.split("\n")
    print("llm split: ", llm_output_beginning_removed_split)

    with open(file,"w") as f:
        for line in llm_output_beginning_removed_split:
            f.write(line)
            f.write("\n")


# parse_solution_and_write_to_file("```python\ndef cumulative_sum(s):\n    '''Yield the cumulative sum of values from iterator s.'''\n    total = 0\n    for value in s:\n        total += value\n        yield total\n```", "lalala.txt")


    
# parse_test_cases(previous_problems)

new_problem = pipeline.run(previous_problems)

print("----------- NEW GENERATED PROBLEM --------------")
with open(output_file, "a") as f:
    f.write("\n")
    f.write(new_problem)
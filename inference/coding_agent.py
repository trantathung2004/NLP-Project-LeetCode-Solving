import os
import textwrap
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_templates import (
    NAIVE_TEMPLATE,
    COT_TEMPLATE
)

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class CodeGenerationAgent:
    def __init__(self, problem_description, samples, model, tokenizer, max_attempts=5):
        self.problem_description = problem_description
        self.samples = samples
        self.max_attempts = max_attempts
        self.model = model
        self.tokenizer = tokenizer
        self.history = [] 

    def run(self):
        feedback = None
        final_code = None

        for attempt in range(1, self.max_attempts + 1):
            print(f"Attempt {attempt}/{self.max_attempts}")
            response = self.generate_code(feedback)
            # Extract the code from the response
            if "```python" in response and "```" in response:
                start_idx = response.find("```python") + len("```python")
                end_idx = response.rfind("```")
                if start_idx < end_idx:
                    generated_code = response[start_idx:end_idx].strip()
            print(f"=== Generated Code (Attempt {attempt}) ===")
            print(generated_code)
            print()
            import code; code.interact(local=locals())
            passed, feedback = self.test_and_feedback(generated_code)
            self.history.append((generated_code, feedback))
            if passed:
                final_code = generated_code
                print("All tests passed!")
                break
            else:
                print("Tests failed, generating feedback for next iteration...\n")

        if final_code:
            print("=== Final Generated Code ===")
            print(final_code)
        else:
            print("Failed to generate passing code within max attempts.")


    def generate_code(self, feedback):
        """
        Call the model to generate or refine code.
        If feedback is None, generate initial code; otherwise, ask to fix.
        """
        if feedback is None:
            prompt = (
                "You are a Python expert. Given the following problem, write a complete Python function or script. "
                "Include definitions for any helper classes (e.g., ListNode), the algorithm should be "
                "implemented in the solve() function and a main() that runs sample tests that called the solve() function.\n\n"
                f"Problem:\n{self.problem_description}\n"
                "Write clear, correct, and efficient code. Your response should be in the following format without any explaination:"
                "```python"
                "YOUR_CODE_HERE"
                "```"
            )
        else:
            prompt = (
                "Your previous code failed on some tests:\n"
                f"{feedback}\n"
                "Please fix the code so that it passes all sample tests. Return the full updated code."
                "Your response should be in the following format without any explaination:"
                "```python"
                "YOUR_CODE_HERE"
                "```"
            )

        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in Python coding."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def test_and_feedback(self, code):
        """
        Execute the generated code, run sample tests, and collect feedback if tests fail.
        Returns (passed: bool, feedback: str).
        """
        # Prepare execution environment
        local_vars = {}
        feedback_msgs = []
        try:
            # Execute the generated code
            exec(code, local_vars, local_vars)

            # Assume solution function is named based on problem; try common names
            # For palindrome linked list: isPalindrome or is_palindrome
            func = None
            for name in ["isPalindrome", "is_palindrome", "solve"]:
                if name in local_vars and callable(local_vars[name]):
                    func = local_vars[name]
                    break

            if func is None:
                return False, "No solution function found."

            # Helper to build linked list if needed
            ListNode = local_vars.get("ListNode", None)
            def build_list(arr):
                if ListNode is None:
                    raise ValueError("ListNode class not defined.")
                dummy = ListNode(0)
                cur = dummy
                for v in arr:
                    cur.next = ListNode(v)
                    cur = cur.next
                return dummy.next

            # Run tests
            for inp, expected in self.samples:
                try:
                    # If input is list for linked list, build it
                    arg = build_list(inp) if isinstance(inp, list) else inp
                    out = func(arg)
                    if out != expected:
                        feedback_msgs.append(
                            f"Input {inp} -> Expected {expected}, Got {out}"
                        )
                except Exception as e:
                    tb = traceback.format_exc()
                    feedback_msgs.append(
                        f"Input {inp} -> Exception: {e}\n{tb}"
                    )

        except Exception as e:
            tb = traceback.format_exc()
            feedback_msgs.append(f"Code execution error: {e}\n{tb}")

        if feedback_msgs:
            feedback = "\n".join(feedback_msgs)
            return False, feedback
        return True, None
    
if __name__ == "__main__":
    # Example problem: Palindrome Linked List
    problem_desc = textwrap.dedent(
        "Given the head of a singly linked list, return true if it is a palindrome or false otherwise. "
        "Follow-up: O(n) time and O(1) space."
    )
    samples = [
        ([1, 2, 2, 1], True),
        ([1, 2], False)
    ]
    agent = CodeGenerationAgent(problem_desc, samples, model, tokenizer)
    agent.run()

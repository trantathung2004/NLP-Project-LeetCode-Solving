import textwrap

NAIVE_TEMPLATE = textwrap.dedent(
    """
    You are given a competitive programming problem. Understand the problem, identify what it is asking, and write a correct and efficient solution in Python.

    Problem:
    {problem_content}

    Write a Python solution that:
    - Parses the input
    - Applies the correct algorithm
    - Prints the correct output for each test case
    Give me the Python implementation only with clear comments.
    """
)

COT_TEMPLATE = textwrap.dedent(
    """
    You are a competitive programming assistant.

    Given the full problem description (including constraints and sample test cases), perform the following steps:

    1. **Understand and summarize the problem** clearly in one or two sentences.
    2. **Identify the input and output format**.
    3. **Analyze constraints and edge cases**, and determine what they imply for algorithm complexity.
    4. **Manually work through sample test cases** and explain the logic and expected output.
    5. **Detect patterns** or structures in the problem that relate to known algorithms or data structures (e.g., Dynamic Programming, Greedy, Graph algorithms, Sliding Window, Binary Search, etc.).
    6. **Devise a correct and efficient algorithm**, step by step, including time and space complexity analysis.
    7. **Implement the algorithm in Python**, ensuring the code is clean, readable, and modular with helpful comments.

    Do not skip any steps. Be thorough, methodical, and explain your reasoning clearly.
    Strictly finish your answer with the implementation in Python like this:
    ```python
    YOUR_SOLUTION
    ```

    Now solve the following problem:
    {problem_content}
    """
)
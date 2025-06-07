# Project Proposal
## Teaching Machines to Code: NLP-Driven LeetCode Problem Solving
### Project Group 2
### April 4, 2025
**1. Problem Description and Significance** 
Solving LeetCode problems involves understanding problem descriptions written in natural language and converting them into correct, executable code. This project explores how
NLP techniques can be used to train a large language model (LLM) to solve such problems efficiently.

Unlike general code generation, LeetCode-style tasks require logic formulation and handling of constraints. We focus on simple methods like prompt engineering and fine-tuning
to test whether they can significantly improve model performance in code synthesis tasks.

This has real-world applications in coding assistants, education, and interview preparation.

**2. Dataset Details** LeetCode Dataset (greengerong)

We use the `greengerong/leetcode` dataset from Hugging Face, which includes problem titles, detailed descriptions, sample test cases, and sometimes reference code. 
Each entry is in JSON format, with fields like title, content, difficulty, and code. Preprocessing steps include markdown removal, whitespace normalization, and converting
entries into prompt-completion pairs.

**Supplementary Datasets**
Additional datasets like CodeSearchNet and APPS may be explored to enrich the model's understanding of syntax and algorithmic structure.

**3. Benchmark Algorithm**
As a baseline, we use retrieval-augmented generation with a pre-trained model (e.g., GPT-2 or CodeGen). This involves retrieving similar problems to guide code synthesis via
contextual prompting.

We compare this baseline against a fine-tuned LLM on the same dataset and explore zero-shot/few-shot prompting with larger models like GPT-3.5 or LLaMA.

**4. Evaluation Metrics**
- Exact Match Accuracy – Does the generated code match the reference solution?
- Test Case Pass Rate – Percentage of sample test cases passed.
- CodeBLEU/BLEU – Measures code similarity.
- Compilation Success – Whether the code runs without errors.

**5. Project Workflow**
1. Dataset Preparation – Load the greengerong/leetcode dataset, filter incomplete entries, and preprocess by normalizing text, removing markdown, and formatting into prompt-completion pairs.
2. Exploratory Analysis – Analyze distributions of difficulty levels, problem lengths, and topics. Visualize patterns and optionally cluster similar problem types.
3. Baseline Modeling – Build a retrieval-augmented prompt generation pipeline using a pre-trained LLM (e.g., GPT-2 or CodeGen). Evaluate performance using few-shot
prompting.
4. Fine-Tuning – Fine-tune a transformer-based model on the preprocessed dataset. Experiment with training parameters and monitor validation loss and early results.
5. Evaluation – Assess model outputs using exact match, test case pass rate, CodeBLEU, and compilation success. Compare against the baseline.
6. Reporting – Summarize findings, document model performance, and highlight failure cases and lessons learned. Prepare final results for presentation or publication.
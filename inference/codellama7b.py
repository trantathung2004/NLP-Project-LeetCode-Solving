import requests
import json
import pandas as pd
from tqdm import tqdm
from data.data_processor import prepare_prompt_with_examples
from pathlib import Path

def get_code_llama_response(prompt, model="codellama:7b-instruct"):
    """
    Get response from CodeLlama model through Ollama API
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"Error: {e}")
        return None

def create_leetcode_prompt(problem_description, train_df, language="python", num_examples=3):
    """
    Create a structured prompt for LeetCode problems with examples
    """
    # Get enhanced prompt with examples
    enhanced_problem = prepare_prompt_with_examples(problem_description, train_df, num_examples)
    
    prompt = f"""You are an expert programming assistant. Please help solve this LeetCode problem:

{enhanced_problem}

Please provide a solution in {language} that:
1. Is efficient and well-commented
2. Includes time and space complexity analysis
3. Explains the approach used
4. Handles edge cases

Solution:"""
    return prompt

def process_test_set(test_file, train_file, output_file="model_results/predictions.csv"):
    """
    Process the test set and save model predictions
    """
    try:
        # Load train and test data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Create results directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Process each test problem
        results = []
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            # print(f"\nProcessing problem {idx + 1}/{len(test_df)}")
            
            # Create prompt with examples
            prompt = create_leetcode_prompt(row['content'], train_df)
            
            # Get model response
            response = get_code_llama_response(prompt)
            
            # Store results
            results.append({
                'problem_id': row['id'],
                'problem_content': row['content'],
                'model_response': response
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing test set: {e}")

def main():
    # File paths
    train_file = "data/split_data/train_set.csv"
    test_file = "data/split_data/test_set.csv"
    output_file = "model_results/predictions.csv"
    
    # Process test set
    process_test_set(test_file, train_file, output_file)

if __name__ == "__main__":
    main()

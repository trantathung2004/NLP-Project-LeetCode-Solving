import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
from pathlib import Path

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def preprocess_code(code):
    """
    Preprocess code for BLEU score calculation
    - Remove comments
    - Remove extra whitespace
    - Tokenize
    """
    if not isinstance(code, str):
        return []
    
    # Remove comments
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove inline comments
        if '#' in line:
            line = line.split('#')[0]
        # Remove empty lines and whitespace
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    # Join and tokenize
    code = ' '.join(cleaned_lines)
    return word_tokenize(code)

def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score between reference and candidate code
    """
    reference_tokens = preprocess_code(reference)
    candidate_tokens = preprocess_code(candidate)
    
    if not reference_tokens or not candidate_tokens:
        return 0.0
    
    # Create reference list for BLEU score
    references = [reference_tokens]
    
    # Calculate BLEU score
    smoothing = SmoothingFunction().method1
    return sentence_bleu(references, candidate_tokens, smoothing_function=smoothing)

def evaluate_predictions(predictions_file, test_file, output_file="model_results/evaluation_results.csv"):
    """
    Evaluate model predictions against test set
    """
    try:
        # Download NLTK data if needed
        download_nltk_data()
        
        # Load predictions and test data
        predictions_df = pd.read_csv(predictions_file)
        test_df = pd.read_csv(test_file)
        
        # Create results directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate BLEU scores
        results = []
        for idx, row in predictions_df.iterrows():
            problem_id = row['problem_id']
            model_response = row['model_response']
            
            # Find corresponding test problem
            test_problem = test_df[test_df['id'] == problem_id].iloc[0]
            reference_solution = test_problem['solution']  # Assuming 'solution' column exists
            
            # Calculate BLEU score
            bleu_score = calculate_bleu_score(reference_solution, model_response)
            
            results.append({
                'problem_id': problem_id,
                'bleu_score': bleu_score,
                'reference_length': len(preprocess_code(reference_solution)),
                'prediction_length': len(preprocess_code(model_response))
            })
        
        # Create evaluation results DataFrame
        eval_df = pd.DataFrame(results)
        
        # Calculate statistics
        stats = {
            'mean_bleu': eval_df['bleu_score'].mean(),
            'median_bleu': eval_df['bleu_score'].median(),
            'std_bleu': eval_df['bleu_score'].std(),
            'min_bleu': eval_df['bleu_score'].min(),
            'max_bleu': eval_df['bleu_score'].max()
        }
        
        # Save detailed results
        eval_df.to_csv(output_file, index=False)
        
        # Print statistics
        print("\nEvaluation Results:")
        print(f"Mean BLEU Score: {stats['mean_bleu']:.4f}")
        print(f"Median BLEU Score: {stats['median_bleu']:.4f}")
        print(f"Standard Deviation: {stats['std_bleu']:.4f}")
        print(f"Min BLEU Score: {stats['min_bleu']:.4f}")
        print(f"Max BLEU Score: {stats['max_bleu']:.4f}")
        
        return stats
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

def main():
    predictions_file = "model_results/predictions.csv"
    test_file = "data/processed_data/split_data/test_set.csv"
    output_file = "model_results/evaluation_results.csv"
    
    evaluate_predictions(predictions_file, test_file, output_file)

if __name__ == "__main__":
    main() 
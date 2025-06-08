import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

def load_and_split_data(input_file, test_size=0.2, random_state=42):
    """
    Load the cleaned data and split it into train and test sets
    """
    try:
        # Read the cleaned CSV file
        df = pd.read_csv(input_file)
        
        # Split the data
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        print(f"Data split complete:")
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        return train_df, test_df
    
    except Exception as e:
        print(f"Error loading and splitting data: {e}")
        return None, None

def save_split_data(train_df, test_df, output_dir):
    """
    Save the split datasets to separate files
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save train and test sets
        train_df.to_csv(output_path / "train_set.csv", index=False)
        test_df.to_csv(output_path / "test_set.csv", index=False)
        
        print(f"Split data saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error saving split data: {e}")

def prepare_prompt_with_examples(problem, train_df, num_examples=3):
    """
    Prepare a prompt that includes example problems from the training set
    """
    # Randomly select example problems
    examples = train_df.sample(n=min(num_examples, len(train_df)))
    
    # Create the examples section
    examples_text = "\nExample Problems:\n"
    for _, row in examples.iterrows():
        examples_text += f"\nProblem {row['id']}:\n{row['content']}\n"
    
    # Combine with the current problem
    full_prompt = f"{examples_text}\nCurrent Problem:\n{problem}"
    
    return full_prompt

def main():
    # File paths
    input_file = "data/leetcode_cleaned.csv"
    output_dir = "data/split_data"
    
    # Load and split data
    train_df, test_df = load_and_split_data(input_file)
    
    if train_df is not None and test_df is not None:
        # Save split data
        save_split_data(train_df, test_df, output_dir)
        
        # Example of preparing a prompt with examples
        if len(test_df) > 0:
            test_problem = test_df.iloc[0]['content']
            enhanced_prompt = prepare_prompt_with_examples(test_problem, train_df)
            print("\nExample of enhanced prompt with training examples:")
            print(enhanced_prompt)

if __name__ == "__main__":
    main() 
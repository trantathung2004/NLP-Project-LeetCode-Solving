import pandas as pd
import re
import json
from datasets import load_dataset
from pathlib import Path

def clean_content(text):
    """
    Clean the content of a LeetCode problem description
    """
    if pd.isna(text):
        return text
    
    # Remove escape sequences
    text = text.replace('\\[', '[').replace('\\]', ']')
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'`(.*?)`', r'\1', text)        # Remove code blocks
    
    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text)              # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n\n', text)       # Replace multiple newlines with double newline
    
    # Fix mathematical notation
    text = re.sub(r'10\^(\d+)', r'10^\1', text)   # Fix power notation
    text = re.sub(r'O\(n\^2\)', 'O(n²)', text)    # Fix big O notation
    
    # Clean up example formatting
    text = re.sub(r'Example\s*(\d+):', r'Example \1:', text)
    
    # Clean up constraints formatting
    text = re.sub(r'Constraints:', '\nConstraints:', text)
    text = re.sub(r'(\d+)\s*<=\s*', r'\1 <= ', text)
    
    return text.strip()

def process_csv(dataset, output_file=None, filename="data/leetcode"):
    """
    Process the CSV file and clean the content column
    """
    try:
        if not isinstance(dataset, pd.DataFrame):
            try:
                df = dataset.to_pandas()
            except:
                print("Cannot convert this data into pd.DataFrame")
        
        if 'content' in df.columns:
            df['content'] = df['content'].apply(clean_content)
        else:
            print("Warning: 'content' column not found in the CSV file")
            return None

        # Save the processed data
        if output_file is None:
            # Create output filename by adding '_cleaned' before the extension
            input_path = Path(filename)
            output_file = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        return df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    # Get LeetCode data from HuggingFace
    ds = load_dataset("greengerong/leetcode", split="train")
    process_csv(ds)

if __name__ == "__main__":
    main()

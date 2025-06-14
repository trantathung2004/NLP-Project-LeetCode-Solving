# LeetCode Problem Solver with CodeLlama

This project uses CodeLlama-7B through Ollama to solve LeetCode problems. It implements a prompting-based approach that leverages example problems to enhance the model's problem-solving capabilities.

## Features

- Data cleaning and preprocessing pipeline
- Train/test split for LeetCode problems
- Enhanced prompting with example problems
- Integration with Ollama's CodeLlama-7B model
- Results tracking and storage

## Prerequisites

- Python 3.8+
- Ollama with CodeLlama-7B model installed
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/trantathung2004/NLP-Project-LeetCode-Solving
cd NLP-Project-LeetCode-Solving
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is running with CodeLlama-7B model:
```bash
ollama run codellama:7b-instruct
```

## Usage
1. Clean the data:
```bash
python data/data_cleaning.py
```

2. Process the data:
```bash
python data/data_processor.py
```

3. Run the main script:
```bash
python main.py
```

## Project Structure

```
.
├── data/
│   ├── data_cleaning.py
│   └── data_processor.py
├── model_results/
│   └── predictions.csv
├── main.py
├── requirements.txt
└── README.md
```

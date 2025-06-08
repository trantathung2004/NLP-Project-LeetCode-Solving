import argparse
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from prompt_templates import (
    NAIVE_TEMPLATE,
    COT_TEMPLATE
)

get_prompt_template = {
    "naive_prompt" : NAIVE_TEMPLATE,
    "cot_prompt" : COT_TEMPLATE
}

def main(model_name, prompting_technique):
    pth_to_test = "data/split_data/test_set.csv"
    test_df = pd.read_csv(pth_to_test)

    PROMPT_TEMPLATE = get_prompt_template[prompting_technique]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        if idx <= 4:
            continue
        problem_content = row['content']
        messages = [
            {"role": "user", "content": PROMPT_TEMPLATE.format(problem_content=problem_content)}
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
        import code; code.interact(local=locals())

def parse_args():
    parser = argparse.ArgumentParser(description="Script to configure prompting techniques and models")

    parser.add_argument(
        "--model_name", 
        choices=["Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"], 
        default="Qwen/Qwen2.5-Coder-3B-Instruct", 
        help="Choose your model"
    )

    parser.add_argument(
        "--prompting_technique", 
        choices=["naive_prompt", "cot_prompt"], 
        default="cot_prompt", 
        help="Choose your prompting technique"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.model_name, args.prompting_technique)
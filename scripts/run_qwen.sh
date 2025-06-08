export PYTHONPATH=$(pwd)

python inference/qwen.py \
    --model_name "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --prompting_technique "cot_prompt"
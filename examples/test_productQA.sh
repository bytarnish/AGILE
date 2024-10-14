# ProductQA & MedMCQA
python3 agile/agent.py \
    --group camera_cases \
    --test_file data/test/camera_cases/qa.jsonl \
    --output_file <output_file> \
    --reflection \
    --seek_advice \
    --use_memory \
    --model vicuna-sft \
    --agent_prompt agile/prompt/agent_for_product_ppo \
    --model_file checkpoints/productqa/agile

# HotPotQa
python3 agile/hotpot_agent.py
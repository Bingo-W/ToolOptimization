python -m optimization.joint_optimization \
    --model_name gpt-4o-mini \
    --save_path chatGPT35_optimization \
    --meta_prompt_path optimization/optimization_prompt \
    --training_data_path data/model_predictions_converted/train_virtual_chatgpt_cot/ \
    --iteration 0 \
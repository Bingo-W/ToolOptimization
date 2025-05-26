#!/usr/bin/env python3

import argparse

from optimization.optimizer import PromptOptimizer

from toolbench.tooleval.evaluators.registered_cls.utils import OpenaiPoolRequest








def main():
    """Main function to run the optimization system."""

    parser = argparse.ArgumentParser()
    # add model name
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help='Model name for OpenAI-based optimizer')
    parser.add_argument('--save_path', type=str, default="chatGPT35_optimization", help='Folder path for optimization files')
    parser.add_argument('--meta_prompt_path', type=str, default="optimization_prompt", help='Path to meta prompt configuration files (without extension)')
    parser.add_argument('--training_data_path', type=str, default="data/model_predictions_converted/train_virtual_chatgpt_cot/", help='Path to training data files')
    parser.add_argument('--iteration', type=int, default=0, help='Current optimization iteration')
    args = parser.parse_args()

    # Configuration
    iteration = 0
    sub_training = 1
    
    # Initialize and run optimizer
    optimizer = PromptOptimizer(
        iteration=iteration,
        sub_training=sub_training,
        folder_path=args.save_path,
        args=args
    )
    
    feedback_dict, optimized_tool_docs = optimizer.run_optimization_pipeline()
    
    print(f"Generated feedback for {len(feedback_dict)} samples")
    print(f"Optimized {len(optimized_tool_docs)} tool descriptions")


if __name__ == "__main__":
    main()
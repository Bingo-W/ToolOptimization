"""
Prompt and Tool Documentation Optimization System

This system implements a three-step optimization process:
1. Feedback Generation - Analyze model predictions and generate feedback
2. Comment Generation - Create improvement suggestions based on feedback
3. Text Updates - Optimize prompts and tool documentation based on suggestions
"""

import json
import os
import re
import argparse
from copy import deepcopy
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from optimization.LLM import OpenAIEngine
from toolbench.tooleval.evaluators.registered_cls.utils import OpenaiPoolRequest


class PromptOptimizer:
    """Main class for the prompt and tool documentation optimization system."""
    
    def __init__(self, iteration: int = 0, sub_training: int = 1, 
                 folder_path: str = "chatGPT35_optimization", args: Optional[argparse.Namespace] = None):
        """
        Initialize the prompt optimizer.
        
        Args:
            iteration: Current optimization iteration
            sub_training: Sub-training iteration
            folder_path: Path to optimization files
        """
        self.iteration = iteration
        self.sub_training = sub_training
        self.folder_path = folder_path
        self.args = args
        
        # create folder if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        # Set up environment
        os.environ["EVAL_MODEL"] = "gpt-4o-mini"
        
        # Load data and configuration
        self.data = self._load_data()
        self.agent_prompt = self._load_agent_prompt()
        self.optimizer_engines = OpenAIEngine(self.args.meta_prompt_path)
        
    def _load_data(self) -> Dict:
        """Load the converted data based on iteration."""
        convert_answer_path = self.args.training_data_path
        
        dataset_name = "trainset"
        datafile_path = os.path.join(convert_answer_path, dataset_name + ".json")
        
        with open(datafile_path, "r") as f:
            return json.load(f)
    
    def _load_agent_prompt(self) -> str:
        """Load the agent prompt based on iteration."""
        if self.iteration == 0:
            return "You should use functions to help handle the real time user querys."
        else:
            file_name = f"optimized_task_description_iter_{self.iteration-1}.txt"
            with open(os.path.join(self.folder_path, file_name), "r") as f:
                return f.read()
    
    @staticmethod
    def create_tool_description(available_tool: List, return_dict: bool = False) -> str:
        """
        Create tool descriptions from available tools.
        
        Args:
            available_tool: List of available tools
            return_dict: Whether to return as dictionary
            
        Returns:
            JSON string or dictionary of tool descriptions
        """
        all_tool_description = {}
        for tool in available_tool:
            if tool['function']['name'] == 'Finish':
                continue
            name = tool['function']['name']
            description = tool['function']['description']
            all_tool_description[name] = description

        return json.dumps(all_tool_description) if not return_dict else all_tool_description
    
    def generate_feedback_and_comments(self) -> Dict:
        """
        Step 1 & 2: Generate feedback and comments for improvement.
        
        Returns:
            Dictionary containing feedback and comments for each sample
        """
        save_path = os.path.join(self.folder_path, f"feedback_dict_iter_{self.iteration}.json")
        
        # Load existing feedback if available
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                feedback_dict = json.load(f)
        else:
            feedback_dict = {}

        print("Generating feedback and comments...")
        for id in tqdm(self.data.keys()):
            if id in feedback_dict.keys() and 'feedback' in feedback_dict[id].keys():
                continue
                
            sample = self.data[id]
            feedback_dict[id] = {}

            try:
                # Step 1: Generate feedback
                query = sample['query']
                raw_answer = sample['answer'].copy()
                raw_answer.pop('method', None)  # Use pop with default to avoid KeyError
                answer = json.dumps(raw_answer)

                feedback = self.optimizer_engines.function_call(
                    'generate_feedback', 
                    {'query': query, 'answer': answer}
                )

                # Load conversational history
                answer_path = self._get_answer_path(id)
                with open(answer_path, "r") as f:
                    conversational_history_data = json.load(f)

                conversational_history = json.dumps(
                    conversational_history_data['answer_generation']['train_messages'][-1]
                )

                # Create tool documentation
                tool_documentations = self.create_tool_description(sample['available_tools'])

                # Step 2: Generate comments
                comments = self.optimizer_engines.function_call(
                    'generate_comment', 
                    {
                        'feedback': feedback['feedback'],
                        'conversation_history': conversational_history,
                        'task_description': self.agent_prompt,
                        'tool_descriptions': tool_documentations
                    }
                )

                feedback_dict[id]['feedback'] = feedback['feedback']
                feedback_dict[id]['comment_for_task_description'] = comments['comment_for_task_description']
                feedback_dict[id]['comment_for_tool_descriptions'] = comments['comment_for_tool_descriptions']

            except Exception as e:
                print(f"Error processing {id}: {e}")
                continue

        # Save feedback dictionary
        with open(save_path, "w") as f:
            json.dump(feedback_dict, f, indent=4)
            
        return feedback_dict
    
    def _get_answer_path(self, id: str) -> str:
        """Get the path to the answer file for a given ID."""
        base_path = self.args.training_data_path
        
            
        answer_path = os.path.join(base_path, f"trainset/{id}_CoT@1.json")
        answer_path = answer_path.replace("model_predictions_converted", "answer")
        return answer_path
    
    def optimize_tool_documentation(self, feedback_dict: Dict, 
                                  finished_ids: Optional[List] = None) -> Dict:
        """
        Step 3: Update and optimize tool documentation.
        
        Args:
            feedback_dict: Dictionary containing feedback and comments
            finished_ids: List of already processed IDs
            
        Returns:
            Dictionary of optimized tool documentations
        """
        if finished_ids is None:
            finished_ids = []
            
        save_path = os.path.join(
            self.folder_path, 
            f"optimized_tool_description_iter_{self.iteration}_sub_{self.sub_training}.json"
        )
        
        # Load existing optimized tool docs if available
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                all_optimized_tool_docs = json.load(f)
        else:
            all_optimized_tool_docs = {}

        # Determine which data to process
        if self.sub_training < 1:
            # Note: sub_training_idxs would need to be defined elsewhere
            data_idx = getattr(self, 'sub_training_idxs', list(self.data.keys()))
        else:
            data_idx = list(self.data.keys())

        print("Optimizing tool documentation...")
        for id in tqdm(data_idx):
            if id in finished_ids or id not in feedback_dict:
                continue

            sample = self.data[id]
            tool_documentations = self.create_tool_description(sample['available_tools'])

            try:
                optimized_tool_description = self.optimizer_engines.function_call(
                    'optimize_text', 
                    {   
                        'text': tool_documentations,
                        'improvement_suggestion': feedback_dict[id]['comment_for_tool_descriptions'],
                    }
                )

                # Parse and save optimized tool description
                optimized_tool_docs_js = json.loads(optimized_tool_description['optimized_context'])
                all_optimized_tool_docs.update(optimized_tool_docs_js)
                finished_ids.append(id)
                
            except Exception as e:
                print(f"Error optimizing tools for {id}: {e}")
                continue

        # Save optimized tool documentation
        with open(save_path, "w") as f:
            json.dump(all_optimized_tool_docs, f, indent=4)
            
        return all_optimized_tool_docs
    
    def run_optimization_pipeline(self) -> tuple:
        """
        Run the complete optimization pipeline.
        
        Returns:
            Tuple of (feedback_dict, optimized_tool_docs)
        """
        print(f"Starting optimization pipeline - Iteration {self.iteration}, Sub-training {self.sub_training}")
        
        # Step 1 & 2: Generate feedback and comments
        feedback_dict = self.generate_feedback_and_comments()
        
        # Step 3: Optimize tool documentation
        optimized_tool_docs = self.optimize_tool_documentation(feedback_dict)
        
        print("Optimization pipeline completed successfully!")
        return feedback_dict, optimized_tool_docs
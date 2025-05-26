"""
Cost-Aware Pass Rate Metric

This module implements a cost-aware pass rate metric that evaluates the performance
of different methods while considering the cost constraint based on tool calling length.
The metric transforms traditional pass rates to account for the efficiency of tool usage.

Author: [Your Name]
Date: [Current Date]
"""

import json
import os
import math
import argparse
from typing import Dict, List, Union
import numpy as np


class CostAwarePassRateCalculator:
    """
    A class to compute cost-aware pass rates for different methods and datasets.
    
    The cost-aware pass rate considers both the success rate and the efficiency
    of tool calling sequences, providing a more comprehensive evaluation metric.
    """
    
    def __init__(self, data_root: str = "data"):
        """
        Initialize the calculator with data root directory.
        
        Args:
            data_root (str): Root directory containing the data files
        """
        self.data_root = data_root
    
    def load_query_dataset(self, query_path: str) -> Dict:
        """
        Load the query dataset from JSON file.
        
        Args:
            query_path (str): Path to the query dataset JSON file
            
        Returns:
            Dict: Loaded query dataset
            
        Raises:
            FileNotFoundError: If the query file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        try:
            with open(query_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Query dataset not found at: {query_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in query dataset: {e}")
    
    def load_pass_rate_results(self, method: str, sub_dataset: str) -> Dict:
        """
        Load pass rate evaluation results for a specific method and dataset.
        
        Args:
            method (str): Method name
            sub_dataset (str): Sub-dataset name
            
        Returns:
            Dict: Pass rate evaluation results
        """
        pass_rate_path = os.path.join(
            self.data_root, 
            "pass_rate_results", 
            method, 
            f"{sub_dataset}_{method}.json"
        )
        
        try:
            with open(pass_rate_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pass rate results not found at: {pass_rate_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in pass rate results: {e}")
    
    def extract_reference_counts(self, query_dataset: List[Dict]) -> Dict[int, int]:
        """
        Extract reference query counts from the dataset.
        
        Args:
            query_dataset (List[Dict]): List of query objects
            
        Returns:
            Dict[int, int]: Mapping of query_id to reference count (+1 for finish function)
        """
        qid_count = {}
        for query in query_dataset:
            query_id = query["query_id"]
            reference_count = len(query["relevant APIs"])
            qid_count[query_id] = reference_count + 1  # +1 for finish function
        
        return qid_count
    
    def generate_cost_thresholds(self, limit: int = 20, limit_max: int = 3) -> np.ndarray:
        """
        Generate cost threshold array for evaluation.
        
        Args:
            limit (int): Number of evaluation points
            limit_max (int): Maximum cost multiplier
            
        Returns:
            np.ndarray: Array of cost thresholds
        """
        step_length = (limit_max - 1) / limit
        return np.arange(1, limit_max + step_length, step_length)
    
    def count_passed_evaluations(self, pass_rate_evaluation: Dict) -> int:
        """
        Count the number of passed evaluations.
        
        Args:
            pass_rate_evaluation (Dict): Pass rate evaluation results for a query
            
        Returns:
            int: Number of passed evaluations
        """
        pass_num = 0
        for evaluation_id in pass_rate_evaluation['is_solved']:
            if pass_rate_evaluation['is_solved'][evaluation_id] == "AnswerStatus.Solved":
                pass_num += 1
        return pass_num
    
    def compute_pass_rate_limited_average(
        self, 
        method: str, 
        sub_dataset: str, 
        query_path: str, 
        limit: int = 20, 
        limit_max: int = 3
    ) -> Dict[float, float]:
        """
        Compute the cost-aware pass rate for different cost constraints.
        
        This method evaluates how well a method performs under various cost constraints,
        where cost is defined as the ratio of actual queries to reference queries.
        
        Args:
            method (str): Method name to evaluate
            sub_dataset (str): Sub-dataset name
            query_path (str): Path to the query dataset JSON file
            limit (int): Number of evaluation points (default: 20)
            limit_max (int): Maximum cost multiplier (default: 3)
            
        Returns:
            Dict[float, float]: Dictionary mapping cost thresholds to pass rates
            
        Raises:
            FileNotFoundError: If required files are not found
            json.JSONDecodeError: If JSON files are malformed
        """
        # Load datasets
        query_dataset = self.load_query_dataset(query_path)
        pass_rate_results = self.load_pass_rate_results(method, sub_dataset)
        
        # Extract reference counts
        qid_count = self.extract_reference_counts(query_dataset)
        
        # Get answer files
        folder_path = os.path.join(self.data_root, "answer", method, sub_dataset)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Answer folder not found: {folder_path}")
        
        all_answer_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        # Initialize results tracking
        cost_thresholds = self.generate_cost_thresholds(limit, limit_max)
        pass_time = {threshold: 0 for threshold in cost_thresholds}
        
        # Process each answer file
        for answer_file in all_answer_files:
            try:
                query_id = int(answer_file.split("_")[0])
            except (ValueError, IndexError):
                print(f"Warning: Could not extract query_id from {answer_file}")
                continue
            
            # Load answer data
            answer_file_path = os.path.join(folder_path, answer_file)
            try:
                with open(answer_file_path, "r", encoding='utf-8') as f:
                    answer = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load {answer_file}: {e}")
                continue
            
            # Extract query count and reference count
            query_count = answer.get('answer_generation', {}).get('query_count', 0)
            reference_count = qid_count.get(query_id, 1)
            
            # Get pass rate evaluation
            query_id_str = str(query_id)
            if query_id_str not in pass_rate_results:
                print(f"Warning: No pass rate results for query_id {query_id}")
                continue
            
            pass_rate_evaluation = pass_rate_results[query_id_str]
            pass_num = self.count_passed_evaluations(pass_rate_evaluation)
            
            # Update pass counts for each cost threshold
            for threshold in cost_thresholds:
                if query_count <= reference_count * threshold:
                    pass_time[threshold] += pass_num
        
        # Calculate final pass rates
        example_num = len(all_answer_files)
        if example_num == 0:
            raise ValueError("No answer files found")
        
        # Get evaluation count from the last processed evaluation
        evaluation_num = len(pass_rate_evaluation.get('is_solved', {}))
        if evaluation_num == 0:
            raise ValueError("No evaluations found")
        
        total_evaluations = example_num * evaluation_num
        pass_rate_limited = {
            threshold: pass_time[threshold] / total_evaluations
            for threshold in cost_thresholds
        }
        
        return pass_rate_limited
    
    def compute_multiple_methods(
        self,
        methods: List[str],
        sub_dataset: str,
        query_path: str,
        limit: int = 20,
        limit_max: int = 3
    ) -> Dict[str, Dict[float, float]]:
        """
        Compute cost-aware pass rates for multiple methods.
        
        Args:
            methods (List[str]): List of method names
            sub_dataset (str): Sub-dataset name
            query_path (str): Path to query dataset
            limit (int): Number of evaluation points
            limit_max (int): Maximum cost multiplier
            
        Returns:
            Dict[str, Dict[float, float]]: Nested dictionary with method -> threshold -> pass_rate
        """
        all_pass_rate_limited = {}
        
        for method in methods:
            try:
                pass_rate_limited = self.compute_pass_rate_limited_average(
                    method=method,
                    sub_dataset=sub_dataset,
                    query_path=query_path,
                    limit=limit,
                    limit_max=limit_max
                )
                all_pass_rate_limited[method] = pass_rate_limited
                print(f"Successfully computed pass rates for method: {method}")
            except Exception as e:
                print(f"Error computing pass rates for method {method}: {e}")
                all_pass_rate_limited[method] = {}
        
        return all_pass_rate_limited
    
    def compute_expectations(self, all_pass_rate_limited: Dict[str, Dict[float, float]]) -> Dict[str, float]:
        """
        Compute expectation (average) of pass rates across all cost thresholds for each method.
        
        Args:
            all_pass_rate_limited (Dict[str, Dict[float, float]]): Pass rates for all methods
            
        Returns:
            Dict[str, float]: Method name to average pass rate mapping
        """
        expectations = {}
        
        for method, pass_rates in all_pass_rate_limited.items():
            if pass_rates:
                expectations[method] = np.mean(list(pass_rates.values()))
            else:
                expectations[method] = 0.0
        
        return expectations

    def print_results_table(self, expectations: Dict[str, float], dataset: str) -> None:
        """
        Print results in a formatted table.
        
        Args:
            expectations (Dict[str, float]): Method expectations to print
        """
        print("\n" + "="*50)
        print("COST-AWARE PASS RATE RESULTS ON {}".format(dataset.upper()))
        print("="*50)
        print(f"{'Method':<30} {'Average Pass Rate':<15}")
        print("-"*50)
        
        for method, expectation in expectations.items():
            print(f"{method:<30} {expectation:<15.4f}")
        
        print("="*50)


def main(args: argparse.Namespace) -> Union[Dict[str, Dict[float, float]], Dict[str, float]]:
    """
    Main function to demonstrate usage of the CostAwarePassRateCalculator.
    """
    # Configuration
    sub_dataset = args.dataset
    query_path = args.query_path
    methods = args.methods
    limit = args.limit
    limit_max = args.limit_max

    # Initialize calculator
    calculator = CostAwarePassRateCalculator()

    try:
        # Compute pass rates for all methods
        print("Computing cost-aware pass rates...")
        all_pass_rate_limited = calculator.compute_multiple_methods(
            methods=methods,
            sub_dataset=sub_dataset,
            query_path=query_path,
            limit=limit,
            limit_max=limit_max
        )
        
        # Compute expectations
        expectations = calculator.compute_expectations(all_pass_rate_limited)

        # Print results
        calculator.print_results_table(expectations, sub_dataset)

        # Return results for further processing if needed
        return all_pass_rate_limited, expectations
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None, None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute Cost-Aware Pass Rate")
    parser.add_argument("--dataset", type=str, default="tooltest_G1_query", help="dataset name")
    parser.add_argument("--query_path", type=str, default="solvable_queries/tool_test_instruction/tooltest_G3_query.json", help="Path to query dataset")
    parser.add_argument("--methods", nargs='+', default=[
        'test_virtual_chatgpt_dfs', 
    ], help="List of methods to evaluate")
    parser.add_argument("--limit", type=int, default=10, help="Number of evaluation points")
    parser.add_argument("--limit_max", type=int, default=5, help="Maximum cost multiplier")
    args = parser.parse_args()
    main(args)
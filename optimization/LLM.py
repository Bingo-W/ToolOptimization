import json
import os
import re
import yaml
from copy import deepcopy
from typing import Dict, Any, List, Optional

from toolbench.tooleval.evaluators.registered_cls.utils import OpenaiPoolRequest


class OpenAIEngine:
    """OpenAI-based engine for generating feedback and optimizing text."""
    
    def __init__(self, cfg_path: str = None):
        """
        Initialize the OpenAI engine with configuration.
        
        Args:
            cfg_path: Path to configuration files (without extension)
        """
        self.opr = OpenaiPoolRequest("openai_key.json")
        self.eval_config = yaml.load(open(cfg_path + '.yaml'), Loader=yaml.FullLoader)
        self.template = open(cfg_path + ".txt").read()
        
        self.parsed_function_templates = {}
        self._parse_function_templates()
        
        self.functions = {}
        for function in self.eval_config['completions_kwargs']['functions']:
            self.functions[function['name']] = function

    def _parse_function_templates(self):
        """Parse function templates from the template string."""
        for function in re.findall(r"<function>(.*?)</function>", self.template, re.DOTALL):
            name = re.findall(r"<name>(.*?)</name>", function, re.DOTALL)[0]
            description = re.findall(r"<description>(.*?)</description>", function, re.DOTALL)[0]
            self.parsed_function_templates[name] = description

    def function_call(self, func_name: str, func_args: Dict, 
                     return_reason: bool = False, return_content: bool = False) -> Dict:
        """
        Call a function using the OpenAI API.
        
        Args:
            func_name: Name of the function to call
            func_args: Arguments for the function
            return_reason: Whether to return the reason
            return_content: Whether to return the content
            
        Returns:
            Dictionary containing the function call results
        """
        completion_kwargs = deepcopy(self.eval_config['completions_kwargs'])
        func_description = deepcopy(self.functions[func_name])

        eval_model = os.getenv('EVAL_MODEL', None)
        if eval_model:
            completion_kwargs['model'] = eval_model

        completion_kwargs.pop('functions')
        completion_kwargs['tools'] = [{'type': 'function', 'function': func_description}]
        completion_kwargs['tool_choice'] = {"type": "function", "function": {"name": func_name}}

        completion_kwargs['messages'] = [{
            'role': 'user',
            'content': str(self.parsed_function_templates[func_name]).format(**func_args)
        }]

        res = self.opr.request(**completion_kwargs)
        ret = json.loads(res.choices[0].message.tool_calls[0].function.arguments)

        required_args = getattr(func_description['parameters'], 'required', None)
        if required_args is not None:
            ret_args = set(ret.keys())
            for arg in required_args:
                if arg not in ret_args:
                    raise KeyError(f"Arg {arg} not found in reply!")

        if return_content:
            ret['content'] = dict(res.choices[0].message).get('content', '')
        
        return ret
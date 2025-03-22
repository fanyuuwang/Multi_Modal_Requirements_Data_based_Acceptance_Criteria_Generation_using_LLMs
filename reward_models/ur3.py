# Absolute Grading: Outputs score of 1 to 5
import json
import os
import sys
import numpy as np
import random
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


class UR3Reranker(torch.nn.Module):
    def __init__(self, model_name, device, quantization_config=None, alpha=0.25):
        super(UR3Reranker, self).__init__()
        self.alpha = alpha
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        if quantization_config is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                          quantization_config=quantization_config).to(self.device)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def compute_log_likelihood(self, prompt_1, prompt_2):
        """Computes the log likelihood of a given text."""
        inputs = self.tokenizer(prompt_1+prompt_2, return_tensors="pt", truncation=True)
        inputs2 = self.tokenizer(prompt_2, return_tensors="pt", truncation=True, add_special_tokens=False)
        eva_length = inputs2["input_ids"].size(1)
        # Move all inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}


        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
        log_likelihood = -F.cross_entropy(
            outputs.logits.squeeze(0)[-eva_length:],
            inputs2['input_ids'].view(-1),
            reduction='mean'
        )
        return log_likelihood.item()

    # def compute_log_likelihood(self, text):
    #     """Computes the log likelihood of a given text."""
    #     inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs, labels=inputs['input_ids'])
    #     log_likelihood = -F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)),
    #                                       inputs['input_ids'].view(-1),
    #                                       reduction='sum')
    #     return log_likelihood.item()

    def prompt_construction(self, ground_knowledge, acceptance_criteria):
        prompt = f"""You are provided with a User Story, Background Knowledge, and its associated Acceptance Criteria. Your task is to evaluate whether the Acceptance Criteria is directly qualified to test the User Story and Background Knowledge based on the following rubric:

    Evaluation Rubric
      1. Relevance assesses how well the generated acceptance criteria aligns with the given information. Ensure that the acceptance criteria align with the given user story, description, and constraints.
2. Coverage evaluates the ‘completeness’ of the generated acceptance criteria, i.e., the extent to which the acceptance criteria encompasses all relevant aspects of the given information. The acceptance criteria should comprehensively address all relevant aspects of the given information.
3. Correctness evaluates the accuracy of acceptance criteria and its steps.The acceptance criteria should be factually accurate and logically structured.
4. Understandability assesses whether the generated acceptance criteria is clear and comprehensible to the experts, without any redundancies.The generated acceptance criteria should be clear, concise, and free from redundancy.
5. Feasibility evaluates whether the acceptance criteria can be executed with the available resources in the project setup. The acceptance criteria should be practical and executable with the available resources in the project setup."""
        prompt_2 = f"""
        You need to base on the following user story and its description:
        
    **User Story**:
    {ground_knowledge[0]}

    *User Story Description*:
    {ground_knowledge[1]}"""

        prompt3 = f"""
        You need to evaluate the following acceptance criteria based on the given context:
        
        **Acceptance Criteria**:
        {acceptance_criteria}"""

        return prompt, prompt_2, prompt3

    def rerank(self, ground, acceptance_criteria):
        prompt1, prompt2, prompt3 = self.prompt_construction(ground, acceptance_criteria)
        query_likelihood = self.compute_log_likelihood(prompt1, prompt2 + prompt3)
        doc_likelihood = self.compute_log_likelihood(prompt3, prompt3)
        risk_score = -query_likelihood - self.alpha * doc_likelihood

        return risk_score


def initiate_ur3_reranker():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "mistralai/Mistral-Small-24B-Instruct-2501"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranker = UR3Reranker(model_name, device, quantization_config)

    return reranker

import json
import os
import sys
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
import math
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax, log_softmax

import torch
import torch.nn.functional as F


class RewardModel(torch.nn.Module):
    def __init__(self, model_name, quantization_config=None):
        super().__init__()
        if quantization_config is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                          quantization_config=quantization_config)
        self.device = self.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("No")

    def prompt_construction(self, ground_knowledge, acceptance_criteria):
        prompt = f"""You are provided with a User Story, Background Knowledge, and its associated Acceptance Criteria. Your task is to evaluate whether the Acceptance Criteria is directly qualified to test the User Story and Background Knowledge based on the following rubric:

Evaluation Rubric
1. Relevance assesses how well the generated acceptance criteria aligns with the given information. Ensure that the acceptance criteria align with the given user story, description, and constraints.
2. Coverage evaluates the ‘completeness’ of the generated acceptance criteria, i.e., the extent to which the acceptance criteria encompasses all relevant aspects of the given information. The acceptance criteria should comprehensively address all relevant aspects of the given information.
3. Correctness evaluates the accuracy of acceptance criteria and its steps.The acceptance criteria should be factually accurate and logically structured.
4. Understandability assesses whether the generated acceptance criteria is clear and comprehensible to the experts, without any redundancies.The generated acceptance criteria should be clear, concise, and free from redundancy.
5. Feasibility evaluates whether the acceptance criteria can be executed with the available resources in the project setup. The acceptance criteria should be practical and executable with the available resources in the project setup.

Based on the above criteria, respond with only one word:
	•	“Yes” if the Acceptance Criteria successfully meets the rubric and is qualified for the User Story and Background Knowledge.
	•	“No” if it does not meet the rubric or is unqualified for the User Story and Background Knowledge.

You need to base on the following user story and its description:

**User Story**:
{ground_knowledge[0]}

*User Story Description*:
{ground_knowledge[1]}

You need to evaluate the following acceptance criteria based on the given context:

**Acceptance Criteria**:
{acceptance_criteria}

Your Response: """

        return prompt

    def gen_verifier(self, background_knowledge, accept_criteria):
        input_prompt = self.prompt_construction(background_knowledge, accept_criteria)
        input_ids, attention_mask = self.cot_decorator(input_prompt)

        # Concatenate context and candidate

        with torch.no_grad():
            log_perplexity = self.gen_verifier_loss(input_ids, attention_mask)

        return log_perplexity.item()

    def cot_decorator(self, input_prompt):
        prompt_encoded = self.tokenizer(input_prompt, return_tensors="pt", truncation=True, add_special_tokens=False)

        # Move tensors to the model's device
        context_input_ids = prompt_encoded["input_ids"].squeeze(0).to(self.model.device)
        context_attention_mask = prompt_encoded["attention_mask"].squeeze(0).to(self.model.device)

        return context_input_ids, context_attention_mask

    def gen_verifier_loss(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits
        final_token_logits = logits[:, -1, :]
        yes_probability = final_token_logits[:, self.yes_token_id].unsqueeze(-1)
        no_probability = final_token_logits[:, self.no_token_id].unsqueeze(-1)
        probabilities = torch.softmax(torch.concat((yes_probability, no_probability)), dim=0)
        loss = probabilities[0]

        return loss

def initiate_gen_veri():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranker = RewardModel(model_name, quantization_config).to(device)

    return reranker

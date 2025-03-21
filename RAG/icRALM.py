import torch
import numpy as np
import random

from sympy.functions.elementary.tests.test_trigonometric import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM

class ICRALM(torch.nn.Module):
    def __init__(self, labels=None):
        super(ICRALM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model & tokenizer
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.labels = labels

    @torch.inference_mode()
    def evaluate_query_with_docs(self, query: str, docs: list[str], topk: int = 1):
        if not docs:
            return {"chosen_doc_idx": None, "neg_log_likelihood": float('nan'), "ppl": float('nan')}

        doc_logprobs = []

        for doc_id, doc_text in enumerate(docs):
            doc_text = "\n".join(doc_text)
            query_text = "\n".join(query)
            doc_enc = self.tokenizer.encode(doc_text, truncation=True)
            query_enc = self.tokenizer.encode(query_text, add_special_tokens=False)
            input_ids = doc_enc + query_enc
            input_ids_tensor = torch.tensor([input_ids], device=self.device)
            labels = [-100] * len(doc_enc) + query_enc
            labels_tensor = torch.tensor([labels], device=self.device)

            out = self.model(input_ids_tensor, labels=labels_tensor)
            num_query_tokens = len(query_enc)
            neg_log_likelihood = out.loss.item() * num_query_tokens
            doc_logprobs.append(neg_log_likelihood)

            # Delete tensors to free up memory
            del input_ids_tensor, labels_tensor, out
            torch.cuda.empty_cache()

        ranking = sorted(zip(docs, doc_logprobs), key=lambda x: x[1])
        labels = None
        if self.labels is not None:
            labels = sorted(zip(self.labels, doc_logprobs), key=lambda x: x[1])
            labels = [labels[x][0] for x in range(topk)]
        return [ranking[x][0] for x in range(topk)], labels


    def evaluate_rag(self,
            query: str,
            docs: list[str],
            topk: int,
    ):
        """
        Evaluate a list of queries with each query having its own list of retrieved docs.
        Returns a list of results (one per query).
        """
        res = self.evaluate_query_with_docs(
            query=query,
            docs=docs,
            topk=topk,
        )
        return res

    def query(self, user_story, docs, topk=2):
        combined_text = f"""**User Story**:
    {user_story[0]}
    **User Story Description**:
    {user_story[1]}"""
        query_results, labels = self.evaluate_rag(combined_text, docs, topk=topk)
        query_results = ["\n".join(x) for x in query_results]
        return query_results, labels


def initiate_TRAG(labels=None):
    model = ICRALM(labels)
    return model

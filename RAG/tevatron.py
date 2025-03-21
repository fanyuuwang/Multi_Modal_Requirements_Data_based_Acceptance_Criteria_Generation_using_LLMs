from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
import torch.nn as nn
from torch.nn.functional import cosine_similarity



class VRAG(nn.Module):
    def __init__(self, dataset):
        super(VRAG, self).__init__()
        model_name_or_path = "Tevatron/dse-phi3-docmatix-v2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained('Tevatron/dse-phi3-docmatix-v2', trust_remote_code=True)
        self.pic_path = "https://raw.githubusercontent.com/fanyuuwang/esolution/main/pages/"
        self.dataset = dataset

    def weighted_mean_pooling(self, hidden, attention_mask):
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = s / d
        return reps

    @torch.no_grad()
    def get_embedding(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        bs = last_hidden_state.shape[0]
        reps = last_hidden_state[torch.arange(bs, device=last_hidden_state.device), sequence_lengths]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    @torch.no_grad()

    def query_construction(self, user_story, description):

        prompt = f"""Here are (1) software requirements of a system formulated in user story and (2) corresponding additional information, where the user story describes the user behavior and additional information provides background knowledge for the software requirements.

        ** User Story:
        {user_story[0]}

        ** User Story Description:
        {user_story[1]}

        ** Description:
        {description}"""

        return prompt

    def us_query_construction(self, user_story):
        us = "\n".join(user_story[0])
        use = "\n".join(user_story[1])
        prompt = f"""Here is software requirements of a system formulated in user story, where the user story describes the user behavior.

        ** User Story:
        {us}

        ** User Story Description:
        {use}"""

        return prompt

    def load_pics(self):
        pic_names = self.dataset
        passages = []
        for pic_name in pic_names:
            try:
                passages.append(Image.open(BytesIO(requests.get(f'{self.pic_path}{pic_name}').content)).resize((1344, 1344)))
            except:
                print(pic_name)
        return passages, [f'{self.pic_path}{name}' for name in pic_names]

    def get_weighted_average_rank(self, rankings, weights):
        """
        Returns a list of item indices ordered from best (first) to worst (last),
        based on a weighted average of ranks across an arbitrary number of rankings.

        :param rankings: List of rankings, where each ranking is a list of item indices
                         in order from best to worst. Example:
                         [
                           [2, 0, 1],     # ranking1
                           [1, 2, 0],     # ranking2
                           [0, 1, 2],     # ranking3
                           ...
                         ]
        :param weights:  List of floats, giving the weight for each ranking.
                         Example (for the above 3 rankings): [1.0, 2.0, 1.0]
                         Must have the same length as 'rankings'.
        :return:         A single list of item indices (best first -> worst last).
        """
        if len(rankings) != len(weights):
            raise ValueError("Number of rankings must match number of weights.")

        # Collect all items from all rankings
        all_items = set()
        for r in rankings:
            all_items.update(r)

        # Build position lookups
        # position_lookup[i][item] = 1-based rank of 'item' in rankings[i]
        position_lookup = []
        for r in rankings:
            pos_dict = {}
            for idx, item in enumerate(r):
                # Convert to 1-based rank: best = 1, next = 2, etc.
                pos_dict[item] = idx + 1
            position_lookup.append(pos_dict)

        # Calculate total weight
        total_weight = sum(weights)

        # Compute each item's weighted average rank
        avg_rank = {}
        for item in all_items:
            weighted_sum = 0.0
            for i, w in enumerate(weights):
                # If item is guaranteed to be in each ranking, we can directly lookup:
                rank_of_item = position_lookup[i][item]
                weighted_sum += w * rank_of_item

            # Weighted average rank
            avg_rank[item] = weighted_sum / total_weight

        # Sort items by ascending average rank (lowest = best)
        sorted_items = sorted(avg_rank.items(), key=lambda x: x[1])

        # Return only the item indices
        return [item_idx for (item_idx, _) in sorted_items]

    def get_weighted_borda_rank(self, rankings, weights):
        """
        Returns a list of item indices ordered from best (first) to worst (last),
        based on a weighted Borda count for any number of rankings.
        """
        if len(rankings) != len(weights):
            raise ValueError("Number of rankings must match number of weights.")

        # Collect all items
        all_items = set()
        for r in rankings:
            all_items.update(r)

        # Assume each ranking has the same number of items
        N = len(rankings[0])

        # Initialize scores
        scores = {item: 0.0 for item in all_items}

        # For each ranking
        for i, ranking in enumerate(rankings):
            weight = weights[i]
            # For position, item in the ranking
            for pos, item in enumerate(ranking):
                # Borda base score: (N - pos)
                # or (N - pos) if 0-based, or (N - pos + 1) if we want the top to get N instead of N-1
                # Let's use (N - pos) for 0-based indexing
                borda_base = (N - pos)
                # Weighted
                scores[item] += weight * borda_base

        # Sort by total score descending (higher = better)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [item_idx for (item_idx, _) in sorted_items]

    @torch.inference_mode()
    def query(self, user_story, topk=5):
        query0 = self.us_query_construction(user_story)

        INSTRUCTION = "Represent this query for retrieving relevant documents: "
        query0 = INSTRUCTION + query0

        passages, pic_names = self.load_pics()
        # passage_prompts = [f"<|image_{p_num+1}|>\nWhat is shown in this image?</s>" for p_num, psg in enumerate(passages)]
        scores0 = []
        for i in range(0, len(passages), 16):
            current_passages = passages[i:i + 16]
            passage_prompts = [f"<|image_{p_num+1}|>\nWhat is shown in this image?</s>" for p_num, psg in enumerate(current_passages)]

            passage_inputs = self.processor(passage_prompts, images=current_passages, return_tensors="pt", padding="longest",
                                            max_length=4096, truncation=True).to(self.device)

            embeddings_query0 = self.processor([query0], return_tensors="pt", padding="longest", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                output = self.model(**embeddings_query0, return_dict=True, output_hidden_states=True)
            embeddings_query0 = self.get_embedding(output.hidden_states[-1], embeddings_query0["attention_mask"])

            passage_inputs['input_ids'] = passage_inputs['input_ids'].squeeze(0)
            passage_inputs['attention_mask'] = passage_inputs['attention_mask'].squeeze(0)
            passage_inputs['image_sizes'] = passage_inputs['image_sizes'].squeeze(0)
            with torch.no_grad():
                output = self.model(**passage_inputs, return_dict=True, output_hidden_states=True)
            doc_embeddings = self.get_embedding(output.hidden_states[-1], passage_inputs["attention_mask"])

            score0 = cosine_similarity(embeddings_query0, doc_embeddings).tolist()

            scores0 = scores0 + score0

        sorted_reward_lists0 = sorted(range(len(scores0)), key=lambda i: scores0[i], reverse=True)

        # sorted_reward_lists0 = [[index for index, value in sorted_score] for sorted_score in sorted_scores0][0]
        sorted_reward_lists0 = [pic_names[x] for x in sorted_reward_lists0[:topk]]
        return sorted_reward_lists0


def initiate_vrag(dataset):
    model = VRAG(dataset)
    model.eval()

    return model

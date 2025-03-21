import torch
from sentence_transformers import SentenceTransformer, util

class RAGWithSentenceTransformer(torch.nn.Module):
    def __init__(self, labels=None):
        super(RAGWithSentenceTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the Sentence Transformer model for generating embeddings.
        self.sbert_model = SentenceTransformer('all-MiniLM-L12-v2').to(self.device)
        self.labels = labels

    @torch.inference_mode()
    def evaluate_query_with_docs(self, query: str, docs: list[str], topk: int = 1):
        """
        Given a query and a list of document strings, compute the cosine similarity
        between the query and each document. Returns the topk documents (based on similarity)
        as a list of strings.
        """
        if not docs:
            return {"chosen_doc_idx": None, "neg_log_likelihood": float('nan'), "ppl": float('nan')}

        cosine_scores = []
        for doc_id, doc_text in enumerate(docs):
            doc_text = "\n".join(doc_text)
            query_text = "\n".join(query)

            # Compute embeddings for the documents and the query.
            doc_embeddings = self.sbert_model.encode(doc_text, convert_to_tensor=True)
            query_embedding = self.sbert_model.encode(query_text, convert_to_tensor=True)

            # Calculate cosine similarity and select the topk documents.
            cosine_scores.append(util.cos_sim(query_embedding, doc_embeddings).item())
        ranking = sorted(zip(docs, cosine_scores), key=lambda x: x[1])
        labels = None
        if self.labels is not None:
            labels = sorted(zip(self.labels, cosine_scores), key=lambda x: x[1])
            labels = [labels[x][0] for x in range(topk)]
        return [ranking[x][0] for x in range(topk)], labels

    def evaluate_rag(self, query: str, docs: list[str], topk: int):
        """
        Wrapper method for evaluating RAG that returns the top-k relevant documents.
        """
        return self.evaluate_query_with_docs(query=query, docs=docs, topk=topk)

    def query(self, user_story, docs, topk=2):
        """
        Formats the input similar to your original implementation and retrieves
        the top-k documents based on the combined user story and its description.
        Expects user_story to be a list or tuple of two strings.
        """
        combined_text = f"""**User Story**:
{user_story[0]}
**User Story Description**:
{user_story[1]}"""
        query_results, labels = self.evaluate_rag(query=combined_text, docs=docs, topk=topk)
        # If the retrieved documents consist of multiple lines, join them for consistency.
        query_results = ["\n".join(x) for x in query_results]
        return query_results, labels


def initiate_simiRAG(labels=None):
    model = RAGWithSentenceTransformer(labels)
    return model

import ollama
import pypdf
from pathlib import Path
import pickle
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

class ChristRAGChatbot:
    def __init__(self, embedding_path="embeddings/rag_embedding.pkl"):
        load_dotenv()
        self.embedding_path = embedding_path
        self.chunks, self.embeddings = self.load_data(embedding_path)
        self.MESSAGES = [
            {"role": "system", "content": """You are an chatbot for Chist University.\n        sometimes the context may be incomplete or not fully relevant.\n        Please answer the question to the best of your ability based on the context provided. """
            },
        ]
        self.azure_endpoint = "https://cto-m6zbzctf-swedencentral.openai.azure.com/"
        self.azure_model = "gpt-4.1"
        self.azure_deployment = "gpt-4.1"
        self.api_version = "2024-12-01-preview"
        self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data['chunks'], data['embeddings']

    def find_similar_chunks(self, query, top_k=5, filter_metadata=None):
        """
        Hybrid retrieval: filter by metadata, then rank by vector similarity.
        filter_metadata: dict, e.g. {"department": "computer science", "type": "syllabus"}
        """
        # If chunks are dicts, filter by metadata
        filtered = self.chunks
        if filter_metadata:
            def match(chunk):
                if not isinstance(chunk, dict):
                    return True
                meta = chunk.get("metadata", {})
                for k, v in filter_metadata.items():
                    if meta.get(k, "").lower() != v.lower():
                        return False
                return True
            filtered = [c for c in self.chunks if match(c)]
            filtered_embeddings = [self.embeddings[i] for i, c in enumerate(self.chunks) if match(c)]
        else:
            filtered_embeddings = self.embeddings

        if not filtered:
            return []
        query_embedding = np.array(ollama.embed(model='nomic-embed-text', input=query)['embeddings'])
        similarities = [np.dot(query_embedding.flatten(), emb.flatten()) for emb in filtered_embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [filtered[i]["text"] if isinstance(filtered[i], dict) and "text" in filtered[i] else filtered[i] for i in top_indices]

    def reformulate_query(self, query, history=None):
        """
        Reformulate the query for better retrieval (expand acronyms, clarify intent, etc).
        Uses the same Azure model to rewrite the query.
        """
        prompt = f"""
You're a university assistant. Rewrite this query for better retrieval:
Query: {query}
History: {history or ''}
Focus on: course names, fees, syllabus topics, locations.
"""
        client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.subscription_key,
        )
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a helpful assistant that rewrites user queries for better retrieval in a university chatbot."},
                      {"role": "user", "content": prompt}],
            max_completion_tokens=100,
            temperature=0.2,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=self.azure_deployment,
        )
        result = response.choices[0].message.content
        client.close()
        return result

    def generate_response(self, query, context_chunks, stream=False):
        context = "\n\n".join(context_chunks)
        prompt = f"""
Context information:
{context}

*sometimes the context may be incomplete or not fully relevant.*

YOU WILL JUST ANSWER THE QUESTIONS ABOUT CHRIST UNIVERSITY.

QUESTION:
{query}
        """
        client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.subscription_key,
        )
        response = client.chat.completions.create(
            stream=stream,
            messages=self.MESSAGES + [{"role": "user", "content": prompt}],
            max_completion_tokens=800,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=self.azure_deployment,
        )
        if stream:
            full_response = ""
            for update in response:
                if update.choices:
                    content = update.choices[0].delta.content or ""
                    full_response += content
            client.close()
            return full_response
        else:
            result = response.choices[0].message.content
            client.close()
            return result

    def answer(self, query, top_k=5, stream=False, filter_metadata=None, history=None, reformulate=True):
        if reformulate:
            query = self.reformulate_query(query, history)
        context_chunks = self.find_similar_chunks(query, top_k=top_k, filter_metadata=filter_metadata)
        return self.generate_response(query, context_chunks, stream=stream)

# Usage example (for integration):
# chatbot = ChristRAGChatbot()
# answer = chatbot.answer("What are all the cs courses offered?")
# print(answer)

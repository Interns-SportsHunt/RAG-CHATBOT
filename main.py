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

    def find_similar_chunks(self, query, top_k=3):
        query_embedding = np.array(ollama.embed(model='nomic-embed-text', input=query)['embeddings'])
        similarities = [np.dot(query_embedding.flatten(), emb.flatten()) for emb in self.embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]

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

    def answer(self, query, top_k=3, stream=False):
        context_chunks = self.find_similar_chunks(query, top_k=top_k)
        return self.generate_response(query, context_chunks, stream=stream)

# Usage example (for integration):
# chatbot = ChristRAGChatbot()
# answer = chatbot.answer("What is the eligibility for MSc Data Science?")
# print(answer)

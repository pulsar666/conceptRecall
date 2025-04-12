import numpy as np
from openai import OpenAI
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

class PathScoringEngine:
    def __init__(self, openai_client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        self.client = openai_client
        self.embedding_model = embedding_model

    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for a given text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def score_paths(self, paths: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Score each path based on semantic alignment with the query.

        Args:
            paths: List of knowledge paths from traversal engine
            query: Original user query text

        Returns:
            List of paths with added 'score' field, sorted by score
        """
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return []

        scored_paths = []

        for path in paths:
            # Concatenate fields from nodes and justifications from relationships
            step_texts = []
            for step in path.get("steps", []):
                source = step.get("source", "")
                relation = step.get("relation", "")
                target = step.get("target", "")
                step_texts.append(f"{source} → [{relation}] → {target}")

            full_text = "\n".join(step_texts)
            path_embedding = self._embed_text(full_text)

            if path_embedding:
                sim = cosine_similarity([query_embedding], [path_embedding])[0][0]
                scored_paths.append({
                    "path": path,
                    "score": float(sim)
                })

        return sorted(scored_paths, key=lambda x: x["score"], reverse=True)

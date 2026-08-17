from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticRouter:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cheap = "llama-3.1-8b-instant"
        self.strong = "llama-3.3-70b-versatile"
        self.threshold = 0.70
        self.fallback = self.cheap

        self.routes = {
        "cheap": [
            "summarize a document",
            "summarize a research paper",
            "translate text",
            "extract information from a document",
            "extract names and entities",
            "format text",
            "rewrite text",
            "simple question answering",
        ],

        "strong": [
            "perform deep analysis of a research paper",
            "analyze the methodology of a research paper",
            "identify weaknesses in a research methodology",
            "perform complex reasoning",
            "generate complex code",
            "debug complex software",
            "solve a multi-step programming problem",
            "reason through a difficult technical problem",
        ],
    }

        # Create embeddings once
        self.route_embeddings = {
            tier: self.model.encode(tasks, normalize_embeddings=True)
            for tier, tasks in self.routes.items()
        }

    def route(self, task: str) -> str:
        task_embedding = self.model.encode(
            task,
            normalize_embeddings=True
        )

        best_tier = None
        best_score = -1

        for tier, embeddings in self.route_embeddings.items():
            scores = np.dot(embeddings, task_embedding)
            score = np.max(scores)

            if score > best_score:
                best_score = score
                best_tier = tier
    
        if best_score >= self.threshold:
            return self.cheap if best_tier == "cheap" else self.strong

        return self.cheap   # fallback

    


if __name__ == "__main__":
    router = SemanticRouter()

    tasks = [
        "Summarize this research paper",
        "Translate this paragraph to Hindi",
        "Extract the names from this document",
        "Write a Python program to implement binary search",
        "Analyze the methodology of this research paper",
        "Debug this complex asynchronous Python application",
        "Generate a Python code script",
    ]

    for task in tasks:
        print(f"{task} → {router.route(task)}")
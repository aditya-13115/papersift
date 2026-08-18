from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticRouter:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cheap = "openai/gpt-oss-20b"
        self.strong = "openai/gpt-oss-120b"
        self.threshold = 0.60
        self.fallback = self.strong  # fallback to strong model if no match

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
        "simple Python scripting",
        "basic HTML generation",
    ],

        "strong": [
        "deep analysis of a research paper",
        "critical analysis and evaluation of research",
        "compare and evaluate multiple approaches",
        "solve complex reasoning problems",
        "design complex software architecture",
        "design scalable distributed systems",
        "develop complex Python applications",
        "debug complex asynchronous software",
        "solve difficult programming problems",
        "analyze performance bottlenecks and concurrency issues",
        "evaluate machine learning experiments",
        "analyze experimental results and identify confounding factors",
        "design production-ready backend systems",
        "perform root cause analysis of production failures",
        "design complex full-stack systems",
        "evaluate technical claims using evidence",
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

            print(f"{tier}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_tier = tier

        print(f"BEST: {best_tier} | SCORE: {best_score:.4f}")

        if best_score >= self.threshold:
            return self.cheap if best_tier == "cheap" else self.strong

        return self.cheap   # fallback

    

if __name__ == "__main__":
    router = SemanticRouter()

    task = """
    Design a fault-tolerant distributed architecture for an AI system
    serving millions of concurrent users. Analyze bottlenecks, failure
    modes, retries, circuit breakers, caching, load balancing, and
    trade-offs between latency, reliability, cost, and consistency.
    """

    print(router.route(task))
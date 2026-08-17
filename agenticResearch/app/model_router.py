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

            if score > best_score:
                best_score = score
                best_tier = tier
    
        if best_score >= self.threshold:
            return self.cheap if best_tier == "cheap" else self.strong

        return self.cheap   # fallback

    


if __name__ == "__main__":
    router = SemanticRouter()

    tasks = [
        # -------------------------
        # Cheap / simple
        # -------------------------
        "Summarize this research paper",
        "Give me a short summary of this document",
        "Translate this paragraph to Hindi",
        "Translate this email into Spanish",
        "Extract the names from this document",
        "Extract all dates and locations from this report",
        "Format this text as a clean Markdown document",
        "Rewrite this paragraph to sound more professional",
        "What is the capital of France?",
        "Convert this list into JSON",
        "Summarize the key points from this meeting",
        "Explain this error message in simple terms",
        "Create a basic HTML page for a portfolio",
        "Write a simple Python script to rename files",

        # -------------------------
        # Strong / complex
        # -------------------------
        "Analyze the methodology of this research paper",
        "Identify weaknesses in this research methodology",
        "Compare the experimental design of these two papers",
        "Critically evaluate the claims made in this research paper",
        "Develop a detailed Python implementation of a distributed task scheduler",
        "Debug this asynchronous Python application",
        "Find the root cause of this race condition",
        "Design an efficient algorithm for this optimization problem",
        "Reason through this system architecture and propose a scalable redesign",
        "Analyze these experimental results and identify confounding factors",
        "Design a production-ready FastAPI service with retries and circuit breakers",
        "Analyze this code for performance bottlenecks and concurrency bugs",
        "Design a scalable distributed system for millions of users",
        "Design a robust agentic system with dynamic tool discovery",
        "Evaluate whether the conclusions of this machine learning experiment are justified",
        "Design a scalable full-stack architecture for a high-traffic application",
        "Solve this complex multi-step programming problem",
        "Design an architecture for a fault-tolerant backend system",

        # -------------------------
        # Boundary / ambiguous
        # -------------------------
        "Explain this research paper in simple terms",
        "Explain why the methodology of this paper may be flawed",
        "Write a Python program to implement binary search",
        "Write a Python program for a concurrent pipeline",
        "Summarize this code and explain what it does",
        "Analyze this code and suggest improvements",
        "Give me an overview of this machine learning paper",
        "Evaluate this machine learning paper",
        "Help me understand this production error",
        "Find the root cause of this production failure",
        "Create a portfolio website",
        "Design a scalable architecture for a portfolio website",
    ]

    for task in tasks:
        print(f"{task} → {router.route(task)}")
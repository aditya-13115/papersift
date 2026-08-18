from collections import deque


class MetricsStore:
    def __init__(self, max_size: int = 1000):
        self.records = deque(maxlen=max_size)

    def record(self, data: dict):
        self.records.append(data)

    def summary(self):
        records = list(self.records)

        if not records:
            return {
                "total_requests": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "average_latency": 0,
                "models": {},
            }

        model_counts = {}

        for record in records:
            model = record["model"]
            model_counts[model] = model_counts.get(model, 0) + 1

        return {
            "total_requests": len(records),

            "total_prompt_tokens": sum(
                r["prompt_tokens"] for r in records
            ),

            "total_completion_tokens": sum(
                r["completion_tokens"] for r in records
            ),

            "total_tokens": sum(
                r["total_tokens"] for r in records
            ),

            "total_cost": sum(
                r["cost"] for r in records
            ),

            "average_latency": sum(
                r["latency"] for r in records
            ) / len(records),

            "models": model_counts,
        }
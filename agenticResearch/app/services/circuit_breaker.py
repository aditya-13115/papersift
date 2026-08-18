import time

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    async def before_call(self):
        if self.state == "OPEN":

            elapsed = time.monotonic() - self.last_failure_time

            if elapsed < self.recovery_timeout:
                raise RuntimeError("Circuit is OPEN")

            # Recovery timeout passed
            self.state = "HALF_OPEN"

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
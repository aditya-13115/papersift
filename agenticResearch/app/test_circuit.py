from app.services.circuit_breaker import CircuitBreaker
import asyncio



async def main():
    circuit = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5
    )

    print("Initial:", circuit.state)

    # 3 failures
    circuit.record_failure()
    print("After 1 failure:", circuit.state)

    circuit.record_failure()
    print("After 2 failures:", circuit.state)

    circuit.record_failure()
    print("After 3 failures:", circuit.state)

    # Should be OPEN
    try:
        await circuit.before_call()
    except RuntimeError as e:
        print("Blocked:", e)

    # Wait for recovery
    print("Waiting 5 seconds...")
    await asyncio.sleep(5)

    # Force it OPEN again
    circuit.record_failure()
    circuit.record_failure()
    circuit.record_failure()

    print("Open again:", circuit.state)

    await circuit.before_call()
    print("After recovery:", circuit.state)

    # Simulate successful test request
    circuit.record_success()
    print("After success:", circuit.state)


asyncio.run(main())
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ValidationError

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. Tool Result
# ============================================================

class ToolResult(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    latency_ms: float


# ============================================================
# 2. Tool Input Schema
# ============================================================

class WeatherArgs(BaseModel):
    city: str = Field(
        description="The name of the city to get the weather for."
    )


# ============================================================
# 3. Base Tool
# ============================================================

class BaseTool(ABC):

    name: str
    description: str
    args_schema: type[BaseModel]

    # Shared concurrency limit
    semaphore = asyncio.Semaphore(5)

    async def validate(self, data: dict):
        return self.args_schema.model_validate(data)

    async def run(self, data: dict) -> ToolResult:

        start = time.perf_counter()

        logger.info(
            "Tool started | name=%s | arguments=%s",
            self.name,
            data
        )

        try:

            # -----------------------------------------
            # 1. Validate arguments
            # -----------------------------------------

            try:
                validated_data = await self.validate(data)

            except ValidationError as e:

                latency = self._latency(start)

                logger.error(
                    "Tool validation failed | name=%s | error=%s | latency_ms=%.2f",
                    self.name,
                    e,
                    latency
                )

                return ToolResult(
                    success=False,
                    error=str(e),
                    latency_ms=latency,
                )

            # -----------------------------------------
            # 2. Control concurrency
            # -----------------------------------------

            async with self.semaphore:

                # -------------------------------------
                # 3. Timeout
                # -------------------------------------

                try:

                    async with asyncio.timeout(10):

                        result = await self._execute(validated_data)

                except asyncio.TimeoutError:

                    latency = self._latency(start)

                    logger.error(
                        "Tool timeout | name=%s | latency_ms=%.2f",
                        self.name,
                        latency
                    )

                    return ToolResult(
                        success=False,
                        error="Tool execution timed out",
                        latency_ms=latency,
                    )

                except Exception as e:

                    latency = self._latency(start)

                    logger.error(
                        "Tool failed | name=%s | error=%s | latency_ms=%.2f",
                        self.name,
                        e,
                        latency
                    )

                    return ToolResult(
                        success=False,
                        error=str(e),
                        latency_ms=latency,
                    )

            # -----------------------------------------
            # 4. Successful execution
            # -----------------------------------------

            latency = self._latency(start)

            logger.info(
                "Tool completed | name=%s | latency_ms=%.2f",
                self.name,
                latency
            )

            return ToolResult(
                success=True,
                data=result,
                latency_ms=latency,
            )

        except Exception as e:

            # Catch unexpected BaseTool-level failures

            latency = self._latency(start)

            logger.exception(
                "Unexpected BaseTool error | name=%s | latency_ms=%.2f",
                self.name,
                latency
            )

            return ToolResult(
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    @staticmethod
    def _latency(start: float) -> float:

        return round(
            (time.perf_counter() - start) * 1000,
            2
        )

    @abstractmethod
    async def _execute(self, arguments: BaseModel):
        """
        Tool-specific logic goes here.
        """
        pass


# ============================================================
# 4. Actual Weather Tool
# ============================================================

class WeatherTool(BaseTool):

    name = "get_weather"

    description = "Get the current weather for a given city."

    args_schema = WeatherArgs

    async def _execute(self, arguments: WeatherArgs):

        mock_temperatures = {
            "New York": 75,
            "Los Angeles": 85,
            "Chicago": 70,
            "Houston": 90,
            "Phoenix": 100,
        }

        return mock_temperatures.get(
            arguments.city,
            "City not found"
        )


# ============================================================
# 5. Test
# ============================================================
'''
async def main():

    weather_tool = WeatherTool()

    result = await weather_tool.run({
        "city": "Chicago"
    })

    print(result)
    print(result.model_dump())


asyncio.run(main())
'''

# ============================================================
# 6. Test Tools
# ============================================================

class FailingTool(BaseTool):
    name = "failing_tool"
    description = "A tool that deliberately raises an exception."
    args_schema = WeatherArgs

    async def _execute(self, arguments: WeatherArgs):
        raise RuntimeError("Something went wrong inside the tool")


class SlowTool(BaseTool):
    name = "slow_tool"
    description = "A tool that deliberately exceeds the timeout."
    args_schema = WeatherArgs

    async def _execute(self, arguments: WeatherArgs):
        await asyncio.sleep(15)
        return "This should never be returned"


class CrashTool(BaseTool):
    name = "crash_tool"
    description = "A tool that deliberately causes a different exception."
    args_schema = WeatherArgs

    async def _execute(self, arguments: WeatherArgs):
        result = 10 / 0
        return result


# ============================================================
# 7. Tests
# ============================================================

async def main():

    weather_tool = WeatherTool()
    failing_tool = FailingTool()
    slow_tool = SlowTool()
    crash_tool = CrashTool()

    # --------------------------------------------------------
    # TEST 1: Normal execution
    # --------------------------------------------------------

    print("\n========== TEST 1: SUCCESS ==========")

    result = await weather_tool.run({
        "city": "Chicago"
    })

    print(result)


    # --------------------------------------------------------
    # TEST 2: Missing required field
    # --------------------------------------------------------

    print("\n========== TEST 2: VALIDATION ERROR ==========")

    result = await weather_tool.run({})

    print(result)


    # --------------------------------------------------------
    # TEST 3: Unknown city
    # --------------------------------------------------------

    print("\n========== TEST 3: UNKNOWN CITY ==========")

    result = await weather_tool.run({
        "city": "Chennai"
    })

    print(result)


    # --------------------------------------------------------
    # TEST 4: Tool raises RuntimeError
    # --------------------------------------------------------

    print("\n========== TEST 4: TOOL EXCEPTION ==========")

    result = await failing_tool.run({
        "city": "Chicago"
    })

    print(result)


    # --------------------------------------------------------
    # TEST 5: Tool raises ZeroDivisionError
    # --------------------------------------------------------

    print("\n========== TEST 5: DIFFERENT EXCEPTION ==========")

    result = await crash_tool.run({
        "city": "Chicago"
    })

    print(result)


    # --------------------------------------------------------
    # TEST 6: Timeout
    # --------------------------------------------------------

    print("\n========== TEST 6: TIMEOUT ==========")

    result = await slow_tool.run({
        "city": "Chicago"
    })

    print(result)


# asyncio.run(main())



class ConcurrencyTool(BaseTool):

    name = "concurrency_test"

    description = "Used to test semaphore concurrency."

    args_schema = WeatherArgs

    # Number of tools currently executing
    active = 0

    # Maximum active tools observed
    max_active = 0

    async def _execute(self, arguments: WeatherArgs):

        ConcurrencyTool.active += 1

        ConcurrencyTool.max_active = max(
            ConcurrencyTool.max_active,
            ConcurrencyTool.active
        )

        logger.info(
            "EXECUTING | city=%s | active=%d",
            arguments.city,
            ConcurrencyTool.active
        )

        # Simulate a slow tool
        await asyncio.sleep(3)

        ConcurrencyTool.active -= 1

        logger.info(
            "FINISHED | city=%s | active=%d",
            arguments.city,
            ConcurrencyTool.active
        )

        return arguments.city


# ============================================================
# 5. Concurrency Test
# ============================================================

async def main():

    tool = ConcurrencyTool()

    start = time.perf_counter()

    # Launch 10 tool calls simultaneously
    tasks = [
        tool.run({
            "city": f"City-{i}"
        })
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)

    total_time = round(
        (time.perf_counter() - start),
        2
    )

    print("\n========================================")
    print("RESULTS")
    print("========================================")

    for i, result in enumerate(results):
        print(f"Task {i}: {result}")

    print("\n========================================")
    print("CONCURRENCY TEST")
    print("========================================")

    print(
        f"Maximum simultaneous executions: "
        f"{ConcurrencyTool.max_active}"
    )

    print(
        f"Total execution time: "
        f"{total_time} seconds"
    )


asyncio.run(main())
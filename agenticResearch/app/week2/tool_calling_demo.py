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

async def main():

    weather_tool = WeatherTool()

    result = await weather_tool.run({
        "city": "Chicago"
    })

    print(result)
    print(result.model_dump())


asyncio.run(main())
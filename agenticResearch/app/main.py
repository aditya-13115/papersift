from time import perf_counter

from pydantic import BaseModel
from fastapi import FastAPI
import asyncio

app = FastAPI()


class User(BaseModel):
    name: str
    age: int
    college: str


class Marks(BaseModel):
    name: str
    marks: int


class Achievement(BaseModel):
    name: str
    achievement: str


async def get_user_data(user: User):
    await asyncio.sleep(2)
    return {"user": user.model_dump()}


async def get_marks_data(marks: Marks):
    await asyncio.sleep(1)
    return {"marks": marks.model_dump()}


async def get_achievement_data(achievement: Achievement):
    await asyncio.sleep(3)
    return {"achievement": achievement.model_dump()}


@app.post("/user")
async def create_user(
    user: User,
    marks: Marks,
    achievement: Achievement
):
    start_time = perf_counter()

    results = await asyncio.gather(
        get_user_data(user),
        get_marks_data(marks),
        get_achievement_data(achievement)
    )

    latency = perf_counter() - start_time

    return {
        "data": results,
        "time_taken": latency
    }
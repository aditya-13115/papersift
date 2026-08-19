from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    age: int = Field(ge=18)

schema = User.model_json_schema()

print(schema)
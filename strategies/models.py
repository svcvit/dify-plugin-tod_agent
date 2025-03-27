from typing import Optional, List, Dict
from pydantic import BaseModel, field_validator
import json
from dify_plugin.interfaces.agent import AgentModelConfig

class DialogueField(BaseModel):
    name: str
    question: str
    required: bool = True
    value: Optional[str] = None

class DialogueState(BaseModel):
    current_field_index: int = 0
    fields: list[DialogueField]
    completed: bool = False

class TODParams(BaseModel):
    information_schema: str
    query: str
    model: AgentModelConfig
    storage_key: str

    @field_validator("information_schema")
    def validate_information_schema(cls, v):
        schema = json.loads(v)
        if not isinstance(schema, dict) or "fields" not in schema:
            raise ValueError("Invalid schema format")
        return v
    

class MTDParams(BaseModel):
    instruction: str
    query: str
    model: AgentModelConfig
    storage_key: str

class DialogueHistory(BaseModel):
    question: str
    answer: str

class MTDContext(BaseModel):
    instruction: str
    current_question: str = ""
    history: List[DialogueHistory] = []
    completed: bool = False
from typing import Dict, Optional
from pydantic import BaseModel


class MyEnvV4Action(BaseModel):
    message: str


class MyEnvV4Observation(BaseModel):
    delay: int
    resources: Dict[str, int]
    echoed_message: Optional[str] = ""


class StepResult(BaseModel):
    observation: MyEnvV4Observation
    reward: float
    done: bool
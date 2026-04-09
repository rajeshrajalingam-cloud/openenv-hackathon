
from fastapi import FastAPI
from models import MyEnvV4Action
from my_env_v4 import MyEnvV4Env

app = FastAPI()
env = MyEnvV4Env()


@app.get("/")
async def root():
    return {"status": "ok", "message": "OpenEnv Space is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset():
    result = await env.reset()
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
    }


@app.post("/step")
async def step(action: dict):
    result = await env.step(MyEnvV4Action(**action))
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
    }


@app.get("/state")
async def state():
    return {
        "delay": env.delay,
        "resources": env.resources,
        "step_count": env.step_count,
        "max_steps": env.max_steps,
        "last_action": env.last_action,
    }
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
from fastapi import FastAPI
import uvicorn
import os

print("LOADED APP FROM:", os.path.abspath(__file__))

from my_env_v4 import MyEnvV4Env
from models import MyEnvV4Action

app = FastAPI(title="OpenEnv Hackathon")

@app.get("/ping-test")

def ping_test():
    return {"ok": True, "file": __file__}



@app.get("/")
def root():
    return {"message": "OpenEnv server running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------
# Grader Endpoints (CRITICAL)
# -------------------------------

@app.get("/grade/delay_recovery")
async def grade_delay():
    score = await run_task("delay_recovery")
    return {"score": score, "reward": score}


@app.get("/grade/resource_crunch")
async def grade_resource():
    score = await run_task("resource_crunch")
    return {"score": score, "reward": score}


@app.get("/grade/priority_conflict")
async def grade_priority():
    score = await run_task("priority_conflict")
    return {"score": score, "reward": score}


# -------------------------------
# Core runner (reuses your env)
# -------------------------------

async def run_task(task_name: str) -> float:
    os.environ["MY_ENV_V4_TASK"] = task_name
    env = await MyEnvV4Env.from_docker_image(None)

    rewards = []
    last_action = None

    try:
        result = await env.reset()

        for step in range(1, env.max_steps + 1):
            if result.done:
                break

            obs = result.observation

            action_text = decide_action(
                task=task_name,
                delay=obs.delay,
                resources=obs.resources,
                step=step,
                last_action=last_action,
            )

            last_action = action_text
            result = await env.step(MyEnvV4Action(message=action_text))

            rewards.append(result.reward or 0.0)

            if result.done:
                break

        if not rewards:
            return 0.01

        score = sum(rewards) / len(rewards)

        # Clamp to validator-safe range
        return max(0.01, min(0.99, score))

    finally:
        await env.close()


# -------------------------------
#  Same logic as inference.py
# -------------------------------

def decide_action(task: str, delay: int, resources: dict, step: int, last_action):
    total_resources = sum(resources.values())

    if task == "delay_recovery":
        if step <= 2:
            if delay > 6:
                action = "prioritize critical tasks"
            elif total_resources < 5:
                action = "reassign resources"
            else:
                action = "optimize workload"
        elif delay > 4:
            action = "prioritize critical tasks"
        elif total_resources < 6:
            action = "reassign resources"
        else:
            action = "optimize workload"

        if step >= 6 and delay <= 3:
            action = "reassign resources"

    elif task == "resource_crunch":
        if step <= 3:
            if total_resources < 5:
                action = "reassign resources"
            elif delay > 5:
                action = "prioritize critical tasks"
            else:
                action = "optimize workload"
        else:
            if delay > 4 and total_resources >= 5:
                action = "prioritize critical tasks"
            elif total_resources < 6:
                action = "reassign resources"
            else:
                action = "optimize workload"

        if step >= 6 and delay <= 3:
            action = "optimize workload"

    elif task == "priority_conflict":
        if delay > 5:
            action = "prioritize critical tasks"
        elif 3 <= delay <= 5:
            action = "optimize workload"
        else:
            action = "optimize workload"

        if total_resources < 4:
            action = "reassign resources"

    else:
        if delay > 6:
            action = "prioritize critical tasks"
        elif total_resources < 5:
            action = "reassign resources"
        else:
            action = "optimize workload"

    if last_action and action == last_action:
        if action == "optimize workload":
            action = "reassign resources"
        elif action == "reassign resources":
            action = "prioritize critical tasks"
        else:
            action = "optimize workload"

    return action


# -------------------------------
# Entry point
# -------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
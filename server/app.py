from fastapi import FastAPI, HTTPException
import uvicorn
import os

from my_env_v4 import MyEnvV4Env
from models import MyEnvV4Action

app = FastAPI(title="OpenEnv Hackathon", version="1.0.0")


@app.get("/")
def root():
    return {
        "message": "OpenEnv server running",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ping-test")
def ping_test():
    return {
        "ok": True,
        "file": os.path.abspath(__file__),
    }


# -------------------------------
# Grader Endpoints
# -------------------------------

@app.get("/grade/delay_recovery")
async def grade_delay_recovery():
    score = await run_task("delay_recovery")
    return {"score": score, "reward": score}


@app.get("/grade/resource_crunch")
async def grade_resource_crunch():
    score = await run_task("resource_crunch")
    return {"score": score, "reward": score}


@app.get("/grade/priority_conflict")
async def grade_priority_conflict():
    score = await run_task("priority_conflict")
    return {"score": score, "reward": score}


# -------------------------------
# Core Runner
# -------------------------------

async def run_task(task_name: str) -> float:
    os.environ["MY_ENV_V4_TASK"] = task_name
    env = None

    try:
        env = await MyEnvV4Env.from_docker_image(None)
        result = await env.reset()

        rewards = []
        last_action = None

        for step in range(1, env.max_steps + 1):
            if result.done:
                break

            obs = result.observation
            resources = obs.resources if obs.resources else {}

            action_text = decide_action(
                task=task_name,
                delay=obs.delay,
                resources=resources,
                step=step,
                last_action=last_action,
            )

            last_action = action_text
            result = await env.step(MyEnvV4Action(message=action_text))

            reward_value = result.reward if result.reward is not None else 0.0
            rewards.append(float(reward_value))

            if result.done:
                break

        if not rewards:
            return 0.1

        score = sum(rewards) / len(rewards)

        # Clamp to validator-friendly range
        return max(0.0, min(1.0, round(score, 4)))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task grading failed: {str(e)}")

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


# -------------------------------
# Decision Logic
# -------------------------------

def decide_action(task: str, delay: int, resources: dict, step: int, last_action: str | None):
    total_resources = sum(resources.values()) if resources else 0

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

    # Prevent repeating the same action over and over
    if last_action and action == last_action:
        if action == "optimize workload":
            action = "reassign resources"
        elif action == "reassign resources":
            action = "prioritize critical tasks"
        else:
            action = "optimize workload"

    return action


# -------------------------------
# Entry Point
# -------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
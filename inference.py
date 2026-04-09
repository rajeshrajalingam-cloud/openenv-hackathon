import asyncio
import os
from typing import List, Optional

from openai import OpenAI
from my_env_v4 import MyEnvV4Action, MyEnvV4Env


API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = "my_env_v4"

TASKS = ["delay_recovery", "resource_crunch", "priority_conflict"]

MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.1


def log_start(task: str, env: str, model: str):
    print(f"\n[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def decide_action(task: str, delay: int, resources: dict, step: int, last_action: Optional[str]) -> str:
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
        if task == "resource_crunch":
            if action == "reassign resources":
                action = "optimize workload"
            elif action == "optimize workload":
                action = "prioritize critical tasks"
            else:
                action = "reassign resources"
        elif task == "priority_conflict":
            if action == "prioritize critical tasks":
                action = "optimize workload"
            elif action == "optimize workload":
                action = "prioritize critical tasks"
            else:
                action = "optimize workload"
        else:
            if action == "optimize workload":
                action = "reassign resources"
            elif action == "reassign resources":
                action = "prioritize critical tasks"
            elif action == "prioritize critical tasks":
                action = "optimize workload"

    return action


async def run_single_task(task_name: str):
    os.environ["MY_ENV_V4_TASK"] = task_name

    env = await MyEnvV4Env.from_docker_image(None)

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0
    last_action = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
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

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_text,
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        if rewards:
            score = sum(rewards) / len(rewards)

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)


async def main():
    client = OpenAI(api_key=API_KEY)

    for task_name in TASKS:
        await run_single_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import os
from typing import List, Optional

from openai import OpenAI
from my_env_v4 import MyEnvV4Action, MyEnvV4Env


MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = "my_env_v4"

TASKS = ["delay_recovery", "resource_crunch", "priority_conflict"]

MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.1

VALID_ACTIONS = [
    "prioritize critical tasks",
    "optimize workload",
    "reassign resources",
]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_BASE_URL = require_env("API_BASE_URL")
API_KEY = require_env("API_KEY")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


def log_start(task: str, env: str, model: str):
    print(f"\n[START] task={task} env={env} model={model}", flush=True)
    print(f"[ENV CHECK] API_BASE_URL={API_BASE_URL}", flush=True)


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


def fallback_action(delay: int, resources: dict) -> str:
    total_resources = sum(resources.values())

    if delay > 6:
        return "prioritize critical tasks"
    if total_resources < 5:
        return "reassign resources"
    return "optimize workload"


def warmup_llm():
    print("[WARMUP] Forcing proxy LLM call...", flush=True)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Reply with exactly OK."},
            {"role": "user", "content": "OK"},
        ],
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    print(f"[WARMUP] Completed with response={content}", flush=True)


def decide_action_llm(
    task: str,
    delay: int,
    resources: dict,
    step: int,
    last_action: Optional[str],
) -> str:
    prompt = f"""
You are selecting the best next action for a project management simulation.

Task: {task}
Step: {step}
Current delay: {delay}
Current resources: {resources}
Previous action: {last_action if last_action else "none"}

Allowed actions:
- prioritize critical tasks
- optimize workload
- reassign resources

Goal:
Reduce delay, use resources effectively, and maximize reward.

Return exactly one action string from the allowed actions.
Do not explain your answer.
""".strip()

    try:
        print("[DEBUG] Calling LLM via LiteLLM proxy...", flush=True)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise decision agent. Return exactly one allowed action string and nothing else.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        print("[DEBUG] LLM call completed", flush=True)

        action = (response.choices[0].message.content or "").strip()

        if action not in VALID_ACTIONS:
            print(f"[LLM WARNING] Invalid action from model: {action!r}", flush=True)
            return fallback_action(delay, resources)

        return action

    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return fallback_action(delay, resources)


async def run_single_task(task_name: str):
    os.environ["MY_ENV_V4_TASK"] = task_name

    env = await MyEnvV4Env.from_docker_image(None)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    last_action: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation

            action_text = decide_action_llm(
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
    warmup_llm()

    for task_name in TASKS:
        await run_single_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())
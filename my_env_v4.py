import random
import os
from models import MyEnvV4Action, MyEnvV4Observation, StepResult


class MyEnvV4Env:
    def __init__(self):
        self.reward: float = 0.0
        self.done: bool = False

        # Core environment state
        self.delay = 10
        self.resources = {"dev": 3, "qa": 2}
        self.step_count = 0
        self.max_steps = 8
        self.last_action = None
        self.task = "delay_recovery"

    @classmethod
    async def from_docker_image(cls, image_name):
        return cls()

    async def reset(self):
        self.task = os.getenv("MY_ENV_V4_TASK", "delay_recovery").strip().lower()

        self.step_count = 0
        self.last_action = None
        self.reward = 0.0
        self.done = False

        # --- Task 1: Delay Recovery ---
        # Goal: reduce high delay efficiently
        if self.task == "delay_recovery":
            self.delay = random.randint(8, 12)
            self.resources = {
                "dev": random.randint(2, 4),
                "qa": random.randint(1, 3)
            }

        # --- Task 2: Resource Crunch ---
        # Goal: recover from low-capacity resource situation
        elif self.task == "resource_crunch":
            self.delay = random.randint(6, 10)
            self.resources = {
                "dev": random.randint(1, 2),
                "qa": random.randint(1, 2)
            }

        # --- Task 3: Priority Conflict ---
        # Goal: handle moderate delay despite having enough resources
        # Best strategy should emphasize prioritization over reallocation
        elif self.task == "priority_conflict":
            self.delay = random.randint(5, 8)
            self.resources = {
                "dev": random.randint(4, 5),
                "qa": random.randint(3, 4)
            }

        # Fallback
        else:
            self.task = "delay_recovery"
            self.delay = 10
            self.resources = {"dev": 3, "qa": 2}

        return StepResult(
            observation=MyEnvV4Observation(
                delay=self.delay,
                resources=self.resources,
                echoed_message=""
            ),
            reward=self.reward,
            done=self.done
        )

    async def step(self, action: MyEnvV4Action):
        self.step_count += 1
        act = action.message.lower().strip()

        reward = 0.0
        total_resources_before = sum(self.resources.values())

        # --- Base action effects ---
        if "prioritize" in act:
            self.delay -= 2
            reward += 0.20

        elif "reassign" in act:
            self.resources["dev"] += 1
            self.resources["qa"] += 1
            reward += 0.10

        elif "optimize" in act:
            self.delay -= 1
            reward += 0.15

        else:
            reward -= 0.20

        # --- Generic penalties / shaping ---
        # Over-prioritizing when delay is already low is wasteful
        if self.delay <= 3 and "prioritize" in act:
            reward -= 0.30

        # Optimizing when delay is severe may be too weak
        if self.delay > 6 and "optimize" in act:
            reward -= 0.20

        # Optimizing with too few resources is ineffective
        if total_resources_before < 4 and "optimize" in act:
            reward -= 0.20

        # Repeating the exact same action is discouraged
        if self.last_action == act:
            reward -= 0.25

        # --- Task-specific reward logic ---
        if self.task == "delay_recovery":
            # Strongly reward aggressive delay reduction
            if "prioritize" in act and self.delay > 4:
                reward += 0.25
            if "reassign" in act and total_resources_before >= 6:
                reward -= 0.10

        elif self.task == "resource_crunch":
            # Resource shortage scenario -> reassign is best
            if "reassign" in act:
                reward += 0.30
            if "prioritize" in act and total_resources_before < 4:
                reward -= 0.20
            if sum(self.resources.values()) >= 6 and "optimize" in act:
                reward += 0.10

        elif self.task == "priority_conflict":
            # Resources are sufficient; main challenge is focus
            if "prioritize" in act and self.delay > 3:
                reward += 0.25
            if "reassign" in act and total_resources_before > 6:
                reward -= 0.20
            if "optimize" in act and 3 <= self.delay <= 5:
                reward += 0.10

        # Keep delay non-negative
        self.delay = max(self.delay, 0)

        # End condition
        self.done = self.step_count >= self.max_steps or self.delay == 0

        # Clamp reward and normalize to 0..1
        reward = max(min(reward, 1.0), -1.0)
        normalized_reward = (reward + 1) / 2

        self.reward = normalized_reward
        self.last_action = act

        return StepResult(
            observation=MyEnvV4Observation(
                delay=self.delay,
                resources=self.resources,
                echoed_message=action.message
            ),
            reward=self.reward,
            done=self.done
        )

    async def close(self):
        pass
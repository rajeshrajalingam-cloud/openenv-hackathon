from my_env_v4 import MyEnvV4Env, MyEnvV4Action


class Client:
    def __init__(self):
        self.env = None

    async def connect(self):
        self.env = await MyEnvV4Env.from_docker_image(None)

    async def reset(self):
        return await self.env.reset()

    async def step(self, action: dict):
        return await self.env.step(MyEnvV4Action(**action))

    async def close(self):
        await self.env.close()
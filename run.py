from models.A3C_model import *

async def main():
    A3c = A3CAgent()
    await A3c.train_with_threads()
    await A3c.save()
    await A3c.test("./defi_env_A3C_0000025_Actor.h5", "./defi_env_A3C_0000025_Critic.h5")

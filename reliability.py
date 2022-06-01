import evaluate
from models.A3C_model import *

def main():
    A3c = A3CAgent()
    rl_reliability = evaluate.load("rl_reliability", "online")
    A3c.test("./defi_env_A3C_0000025_Actor.h5", "./defi_env_A3C_0000025_Critic.h5")
    results = rl_reliability.compute(
        timesteps=[list(range(A3c.episode))],
        rewards=[A3c.rewards]
        )
    print(results)
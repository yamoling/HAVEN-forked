from envs import shaped_lle
from lle import Action, LLE
import time

env = shaped_lle(gamma=0.95, map=6, enable_shaped_subgoals=True, reward_value=1.0, obs_type="layered")
# env = LLE.level(6).single_objective()
done = False
start = time.time()
for i in range(100_000):
    if i % 1000 == 0:
        print(i)
    if done:
        env.reset()

    action = env.env.action_space.sample(env.env.available_actions())
    _, done, _ = env.step(action)


duration = time.time() - start
duration_ms = duration * 1000
print(f"Duration: {duration}s")
print(f"Avg time per step: {duration_ms / i}ms")

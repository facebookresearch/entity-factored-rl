# Fetch N-Push and N-Switch environments

If your code is outside the envs directory, you can initialize envs like follows:

```python
import envs
import gym

env = gym.make("Fetch3Push-v1")  # 3-Push, sparse reward
env = gym.make("Fetch3PushDense-v1")  # 3-Push, dense reward

env = gym.make("Fetch3Switch-v1")  # 3-Switch, sparse reward

env = gym.make("Fetch2Switch2Push-v1")  # 2 switches and 2 cubes

env = gym.make("Fetch2SwitchOr2Push-v1")  # 2 switches and 2 cubes, but the goal only involves either the switches or the cubes
```

from gym.envs.registration import register

register(
  id='lgsvl-v0',
  entry_point='gym_lgsvl.envs:LgsvlEnv',
  # episode_max_steps=1000,
)

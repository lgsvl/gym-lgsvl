from gym.envs.registration import register

register(
  id='lgsvl-v0',
  entry_point='gym_lgsvl.envs:LgsvlEnv',
  kwargs={},
  max_episode_steps=100,
)

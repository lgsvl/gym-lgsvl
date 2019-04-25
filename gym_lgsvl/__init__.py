from gym.envs.registration import register

register(
  id='lgsvl-v0',
  entry_point='gym_lgsvl.envs:LgsvlEnv',
  kwargs={
    'scene': 'SanFrancisco',
    'port': 8181,
  },
  max_episode_steps=50,
)

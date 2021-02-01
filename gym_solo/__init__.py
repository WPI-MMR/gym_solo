from gym.envs.registration import register

register(
  id='solo8vanilla-v0',
  entry_point='gym_solo.envs:Solo8VanillaEnv',
)

register(
  id='solo8vanilla-realtime-v0',
  entry_point='gym_solo.envs:RealtimeSolo8VanillaEnv',
)
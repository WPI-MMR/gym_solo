"""Custom solo environments"""
# Base (abstract) environment
from gym_solo.envs.solo8_base_env import Solo8BaseEnv

# First party model
from gym_solo.envs.solo8v2vanilla import Solo8VanillaEnv

# First party model realtime simulation
from gym_solo.envs.solo8v2vanilla_realtime import RealtimeSolo8VanillaEnv
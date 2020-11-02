from gym_solo.core import rewards


class SimpleReward(rewards.Reward):
  def compute(self):
    return 1
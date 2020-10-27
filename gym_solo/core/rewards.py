from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


class Reward(ABC):  
  @abstractmethod
  def compute(self) -> solo_types.reward:
    """Compute the reward for the current state.

    Returns:
      solo_types.reward: The reward evalulated at the current state.
    """
    pass


@dataclass
class WeightedReward:
  reward: Reward
  weight: float


class RewardFactory:
  def __init__(self):
    self._rewards: List[WeightedReward] = []

  def register_reward(self, reward, weight):
    self._rewards.append(reward=reward, weight=weight)

  def get_reward(self):
    return sum(wr.weight * wr.reward.compute() for wr in self._rewards)
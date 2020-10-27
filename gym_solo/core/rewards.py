from abc import ABC, abstractmethod

class Reward(ABC):  
  @abstractmethod
  def compute(self) -> solo_types.reward:
    """Compute the reward for the current state.

    Returns:
        solo_types.reward: The reward evalulated at the current state.
    """
    pass
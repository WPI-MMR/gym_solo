from abc import ABC, abstractmethod


class Termination(ABC):
  @abstractmethod
  def reset(self):
    """Resets the state of the termination condition
    """
    pass

  @abstractmethod
  def is_terminated(self) -> bool:
    """This should implement functionanlity that determines when an episode
    should terminate
    """
    pass


class TerminationFactory:
  def __init__(self):
    """Create a new Termination Factory. This way users can add multiple
    different termination conditions and different ways to execute them to
    signal the end of an episode
    """
    self._terminations = []
    self._use_or = True

  def register_termination(self, *terminations):
    """Add a termination condition used to evaluate when an episode
    has ended.

    Args:
      term (Termination): Termination condition used to signal the 
      end of an episode 
    """
    for termination in terminations:
      self._terminations.append(termination)

  def is_terminated(self) -> bool:
    """Tells whether an episode is terminated based on the registered
    termination conditions and value of _is_or attribute
    """
    if self._use_or:
      for termination in self._terminations:
        if termination.is_terminated():
          return True
      return False
    else:
      raise ValueError('No termination condition other than OR is defined')

  def reset(self):
    """Resets all the termination conditions stored
    """
    for termination in self._terminations:
      termination.reset()


class TimeBasedTermination(Termination):
  """The episode terminated when the step_delta becomes greater than
  max_step_delta
  """
  def __init__(self, max_step_delta: int):
    self.max_step_delta = max_step_delta
    self.reset()
  
  def reset(self):
    self.step_delta = 0

  def is_terminated(self) -> bool:
    """Return true when tep_delta becomes greater than max_step_delta.
    Otherwise return false.
    """
    self.step_delta += 1
    return self.step_delta > self.max_step_delta


   
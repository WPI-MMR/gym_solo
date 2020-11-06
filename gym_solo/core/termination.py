from abc import ABC, abstractmethod


class Termination(ABC):
  def __init__(self, body_id: int, *args, **kwargs):
      """This is used to define the termination condition

      Args:
        body_id (int): PyBullet body id to calculate the termination
        condition for.
      """
      self.body_id = body_id
      self.create_args()
      self.reset()
  
  @abstractmethod
  def create_args(self, *args, **kargs):
    """Assigns the arguments to class attributes
    """
    pass

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

  def register_termination(self, term: Termination):
    """Add a termination condittion used to evaluate when an episode
    has ended.

    Args:
      term (Termination): Termination condition used to signal the 
      end of an episode 
    """
    self._terminations.append(term)

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
  """Stratergy to signal end of an episode based on time steps
  """
  def assign_args(self, max_step_delta: int):
    self.max_step_delta = max_step_delta
  
  def reset(self):
    self.step_delta = 0

  def is_terminated(self) -> bool:
    """
    docstring
    """
    self.step_delta += 1
    return self.step_delta > self.max_step_delta


   
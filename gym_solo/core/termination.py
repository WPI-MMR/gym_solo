from abc import ABC, abstractmethod


class Termination(ABC):
    @abstractmethod
    def __init__(self, body_id: int):
        """This is used to define the termination condition

        Args:
          body_id (int): PyBullet body id to calculate the termination
          condition for.
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
      for i in range(len(self._terminations)):
        if self._terminations[i]:
          return True
    else:
      raise ValueError('No termination condition other than OR is defined')


class TimeBasedTermination(Termination):
  """Stratergy to signal end of an episode based on time steps
  """
  def __init__(self, body_id: int):
    """Creates a time based termination condition
    
    Args:
      body_id (int): The PyBullet body id for the robot.
    """
    self.body_id = body_id
    self.reset()

  def reset(self):
    self.time = 0
    self.max_time = 100,000

  def is_terminated(self) -> bool:
    """
    docstring
    """
    if self.time > self.max_time:
      return True
    self.time += 1


   
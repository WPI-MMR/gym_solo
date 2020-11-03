from abc import ABC, abstractmethod


class Termination(ABC):
    @abstractmethod
    def __init__(self):
        """This is used to define the termination condition
        """
        pass
    
    @abstractmethod
    def is_terminated(self):
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
    self._termination = []
    self._is_or = True

  def register_observation(self, term: Termination):
    """Add a termination condittion used to evaluate when an episode
    has ended.

    Args:
      term (Termination): Termination condition used to signal the 
      end of an episode 
    """
    pass

  def register_execution_type_as_or(self, is_or: bool = True):
    """If isOr is true, then all the termination conditions are executed as or
    conditional statements
    """
    pass

  def is_terminated(self) -> bool:
    """Tells whether an episode is terminated based on the registered
    termination conditions and value of _is_or attribute
    """"
    pass

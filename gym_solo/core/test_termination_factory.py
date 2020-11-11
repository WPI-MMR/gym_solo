import unittest
from gym_solo.core import termination

# TODO: Move this to gym_solo.testing
class DummyTermination(termination.Termination):
  def __init__(self, body_id: int, termination_var: bool):
    self.body_id = body_id
    self.termination_var = termination_var
    self.reset_counter = 0
    self.reset()
    
  def reset(self):
    self.reset_counter += 1

  def is_terminated(self) -> bool:
    return self.termination_var

class TestTerminationFactory(unittest.TestCase):
  def test_initialization(self):
    termination_factory = termination.TerminationFactory()
    self.assertFalse(termination_factory._terminations)
    self.assertTrue(termination_factory._use_or)

  def test_register_termination(self):
    termination_factory = termination.TerminationFactory()
    dummy_termination_1 = DummyTermination(0, True)
    dummy_termination_2 = DummyTermination(0, False)
    termination_factory.register_termination(dummy_termination_1, 
      dummy_termination_2)
    self.assertEqual(len(termination_factory._terminations), 2)

  def test_is_terminated(self):
    dummy_termination_true_1 = DummyTermination(0, True)
    dummy_termination_false_1 = DummyTermination(0, False)    
    dummy_termination_false_2 = DummyTermination(0, False)
    
    with self.subTest('True and False Termination'):
      termination_factory = termination.TerminationFactory()
      termination_factory.register_termination(dummy_termination_true_1, 
        dummy_termination_false_1)
      self.assertTrue(termination_factory.is_terminated())    

    with self.subTest('False and False Termination'):
      termination_factory = termination.TerminationFactory()
      termination_factory.register_termination(dummy_termination_false_1, 
        dummy_termination_false_2)
      self.assertFalse(termination_factory.is_terminated())
  
  def test_reset(self):
    dummy_termination = DummyTermination(0, True)
    termination_factory = termination.TerminationFactory()
    termination_factory.register_termination(dummy_termination)
    self.assertEqual(1, dummy_termination.reset_counter)
    termination_factory.reset()
    self.assertEqual(2, dummy_termination.reset_counter)
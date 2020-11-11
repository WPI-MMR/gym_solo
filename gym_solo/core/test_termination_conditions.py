import unittest
from gym_solo.core import termination

class TestTimeBasedTermination(unittest.TestCase):
    def test_attributes(self):
      max_step_delta = 3
      term = termination.TimeBasedTermination(max_step_delta)
      self.assertEqual(max_step_delta, term.max_step_delta)
      self.assertEqual(0, term.step_delta)

    def test_reset(self):
      max_step_delta = 3
      term = termination.TimeBasedTermination(max_step_delta)

      for i in range(max_step_delta):
        term.is_terminated()

      term.reset()
      self.assertEqual(0,term.step_delta)

    def test_is_terminated(self):
      max_step_delta = 3
      term = termination.TimeBasedTermination(max_step_delta)
      
      for i in range(max_step_delta):
        self.assertEqual(False, term.is_terminated())
        self.assertEqual(i+1, term.step_delta)

      self.assertEqual(True, term.is_terminated())
      self.assertEqual(max_step_delta + 1, term.step_delta)
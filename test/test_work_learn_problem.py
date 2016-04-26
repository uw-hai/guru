import unittest
from .. import work_learn_problem as wlp

class TestGenerateStates(unittest.TestCase):
    def setUp(self):
        self.n_skills = 5
        self.n_worker_classes = 3
        self.states = wlp.states_all(n_skills=self.n_skills,
                                     n_worker_classes=self.n_worker_classes)

    def test_len(self):
        n_states = 1 + 2 ** self.n_skills * \
                       (self.n_skills + 1) * self.n_worker_classes
        self.assertEqual(n_states, len(self.states))

if __name__ == '__main__':
    unittest.main()

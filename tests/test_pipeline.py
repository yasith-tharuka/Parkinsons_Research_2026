import unittest
import numpy as np
from src.experiments.run_experiment import choose_operating_threshold

class TestParkinsonsPipeline(unittest.TestCase):
    
    def test_choose_operating_threshold_perfect_case(self):
        """Test threshold selection when there is a perfect classification boundary."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.9, 0.85, 0.95])
        
        result = choose_operating_threshold(y_true, y_prob)
        
        self.assertGreaterEqual(result['threshold'], 0.3)
        self.assertLessEqual(result['threshold'], 0.8)
        self.assertEqual(result['sensitivity'], 1.0)
        self.assertEqual(result['specificity'], 1.0)

    def test_choose_operating_threshold_fallback(self):
        """Test threshold selection under poor probability calibration where fallback is needed."""
        y_true = np.array([0, 0, 1, 1])
        # Model gives very low probabilities to everyone, making 100% sensitivity hard at normal threshold
        y_prob = np.array([0.01, 0.02, 0.05, 0.06])
        
        result = choose_operating_threshold(y_true, y_prob, target_sensitivity=1.0, min_specificity=0.25)
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['sensitivity'], 0.5)

if __name__ == '__main__':
    unittest.main()

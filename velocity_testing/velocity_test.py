import pandas as pd
import numpy as np
from velocity_testing.calculate_velocity import calculate_velocity
from scipy.stats import ttest_ind


class VelocityTester:
    def __init__(self, data, weights, testing_mode):
        self.data = data
        self.testing_mode = testing_mode
        self.weights = weights
        self.velocity_test_result = None
    
    def run_test(self):
        if self.testing_mode == 'ttest':
            self.velocity_test_result = self.ttest()

        return self.velocity_test_result

    def ttest(self, labels, g, n):
        # calculate the velocities for both groups
        velocity_results = []
        for cluster in [g, n]:
            velocity_results.append(
                calculate_velocity(
                    self.data[labels == cluster], 
                    self.weights[labels == cluster]
                )
            )

        # calculate the t-statistic and p-value
        _, pvalues = ttest_ind(velocity_results[0], velocity_results[1], equal_var=False)

        return self.is_same_velocity(pvalues)
    
    @staticmethod
    def is_same_velocity(pvalues):
        return np.all(pvalues > 0.05)
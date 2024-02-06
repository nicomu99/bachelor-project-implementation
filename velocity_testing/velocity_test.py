import pandas as pd
import numpy as np
from velocity_testing.calculate_velocity import calculate_velocity
from scipy.stats import ttest_ind
from scipy.spatial import distance
import sys

sys.path.append('/home/nico/VSCodeRepos/SigMA')
from NoiseRemoval.bulk_velocity_solver_matrix import dense_sample, bootstrap_bulk_velocity_solver_matrix, bulk_velocity_solver_matrix
from NoiseRemoval.OptimalVelocity import vr_solver, transform_velocity, optimize_velocity
from NoiseRemoval.xd_outlier import XDOutlier
from miscellaneous.covariance_trafo_sky2gal import transform_covariance_shper2gal
from miscellaneous.error_sampler import ErrorSampler


class VelocityTester:
    def __init__(self, data, weights, testing_mode, err_sampler=None):
        self.data = data
        self.testing_mode = testing_mode
        self.weights = weights
        self.error_sampler = err_sampler
    
    def run_test(self, labels, g, n, return_stats=False):
        if self.testing_mode   == 'ttest':
            return self.ttest                               (labels, g, n, return_stats)
        elif self.testing_mode == 'bootstrap_range_test':
            return self.bootstrap_range_test                (labels, g, n, return_stats)
        elif self.testing_mode == 'xd_mean':
            return self.xd_mean                             (labels, g, n, return_stats)
        elif self.testing_mode == 'error_sample_ttest':
            return self.error_sample_ttest                  (labels, g, n, return_stats)
        elif self.testing_mode == 'xd_mean_sample_distance':
            return self.xd_mean_sample_distance             (labels, g, n, return_stats)
        elif self.testing_mode == 'error_sample_bootstrap_range_test':
            return self.error_sample_bootstrap_range_test   (labels, g, n, return_stats)
        else:
            raise ValueError("Invalid testing mode")


    def ttest(self, labels, cluster1, cluster2, return_stats=False):
        """
        Perform a t-test on the velocities of two clusters

        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        cluster1 : int
            Label of the first cluster
        cluster2 : int
            Label of the second cluster
        return_stats : bool
            Return the statistics of the test

        Returns
        -------
        bool
            True if the velocities are the same, False otherwise
        """
        # calculate the velocities for both clusters
        velocity_results = [
            calculate_velocity(
                self.data[labels == cluster], 
                self.weights[labels == cluster]
            )
            for cluster in [cluster1, cluster2]
        ]

        # calculate the t-statistic and p-value
        t_stat, pvalues = ttest_ind(*velocity_results, equal_var=False)

        if return_stats:
            return self.is_same_velocity(pvalues), velocity_results, {'pvalues': pvalues, 'stats':t_stat}
        return self.is_same_velocity(pvalues)
    
    def bootstrap_difference_test(self, labels, cluster1, cluster2, return_stats=False):
        """
        Perform a two-sided bootstrap test.
        
        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        cluster1 : int
            Label of the first cluster
        cluster2 : int
            Label of the second cluster
        return_stats : bool
        
        Returns
        -------
        bool
            True if the velocities are the same, False otherwise"""

        n_bootstraps = 100

        # calculate the true velocities for both groups
        velocities_true = [
            bulk_velocity_solver_matrix(
                self.data[labels == cluster],
                self.weights[labels == cluster]
            )
            for cluster in [cluster1, cluster2]
        ]

        # calculate the difference
        velocity_differences = velocities_true[0] - velocities_true[1]

        # calculate the velocities for both groups
        velocity_bootstrap = [
            bootstrap_bulk_velocity_solver_matrix(
                self.data[labels == cluster], 
                self.weights[labels == cluster],
                n_bootstraps=n_bootstraps
            )
            for cluster in [cluster1, cluster2]
        ]

        # calculate the difference between each bootstrap
        velocity_differences_bootstrap = [
            velocity_bootstrap[0][i] - velocity_bootstrap[1][i]
            for i in range(50)
        ]

        pvalue = np.max([
            ((velocity_differences_bootstrap > velocity_differences) + 1) / (n_bootstraps + 1), 
            ((velocity_differences_bootstrap < velocity_differences) + 1) / (n_bootstraps + 1)
        ])
        pvalue *= 2

        # the pvalue has to be smaller than 0.05 to reject the null hypothesis
        if return_stats:
            return pvalue < 0.05, velocity_differences_bootstrap, pvalue
        return pvalue < 0.05
    
    @staticmethod
    def is_same_velocity(pvalues):
        return np.all(pvalues > 0.05)
    
    def get_xd(self, cluster_index):
        # Written by Sebastian Ratzenböck
        # get dense sample
        dense_core = dense_sample(self.weights[cluster_index])

        c_vel = ['U', 'V', 'W']
        X = self.data[cluster_index].loc[dense_core, c_vel].values
        C = self.error_sampler.C[cluster_index][dense_core, 3:, 3:]

        ra, dec, plx = self.data[cluster_index].loc[dense_core, ['ra', 'dec', 'parallax']].values.T
        # Compute covariance matrix in Galactic coordinates
        C_uvw = transform_covariance_shper2gal(ra, dec, plx, C)     
        xd = XDOutlier().fit(X, Xerr)
        return xd
    
    @staticmethod
    def calculate_distance(self, V, mu, x):
        try:
            # reshape and invert the cov matrix
            iv = np.linalg.inv(V.reshape(3, 3))
        except:
            raise ValueError("Covariance Matrix can not be inverted")

        # flatten the input arrays
        mu = mu.flatten()
        x = x.flatten()

        return distance.mahalanobis(x, mu, iv)
    
    def xd_mean(self, labels, g, n, return_stats=False):
        # for each cluster, get the xd object
        xd = []
        for cluster in [g, n]:
            cluster_index = labels == cluster
            xd.append(self.get_xd(cluster_index))

        # get the mahalanobis distance between the two clusters and maximize it
        max_mahalanobis_distance = max(
            self.calculate_distance(xd[0].V, xd[0].mu, xd[1].mu),
            self.calculate_distance(xd[1].V, xd[1].mu, xd[0].mu)
        )

        return max_mahalanobis_distance < 2, [xd[0].mu, xd[1].mu], max_mahalanobis_distance
    
        
    def get_error_sample(self, data):
        # generate a new sampler on the cluster data
        err_sampler = ErrorSampler(data)
        err_sampler.build_covariance_matrix()
        # Create sample from errors
        cols = ['X', 'Y', 'Z', 'U', 'V', 'W']
        data_new = pd.DataFrame(err_sampler.spher2cart(err_sampler.new_sample()), columns=cols)

        return data_new[['U', 'V', 'W']]
    
    def error_sample_ttest(self, labels, g, n, return_stats=False):
        # get the error sample for both clusters
        err_sample = []
        for cluster in [g, n]:
            cluster_index = labels == cluster
            err_sample.append(self.get_error_sample(cluster_index))

        # calculate a t-test on the error samples
        _, pvalues = ttest_ind(err_sample[0], err_sample[1], equal_var=False)

        if return_stats:
            return self.is_same_velocity(pvalues), err_sample, pvalues
        return self.is_same_velocity(pvalues), err_sample
    
    def xd_mean_sample_distance(self, labels, g, n, return_stats=False):
        # we need the error sampler for both clusters to generate new samples
        # we need the extreme deconvolution to calculate the mahalanobis distance
        # we then calculate the maximal distance from each cluster to the mean of the other cluster
        # and finally take the minimum of those two distances

        samples, xd = [], []
        for cluster in [g, n]:
            cluster_index = labels == cluster
            samples.append(self.get_error_sample(cluster_index))
            xd.append(self.get_xd(self.error_sampler, cluster_index))

        # calculate the maximal distance between each point in the samples and the mean of the other cluster
        distances = []
        for i in range(2):
            distances.append(
                np.max(
                    np.array(
                        [
                            self.calculate_distance(xd[i].V, xd[i].mu, sample)
                            for sample in samples[1 - i].values
                        ]
                    )
                )
            )

        # take the minimum of the two distances
        min_distance = min(distances)

        return min_distance < 2, [xd[0].mu, xd[1].mu], min_distance
    
    def error_sample_bootstrap_range_test(self, labels, g, n, return_stats):
        # we bootstrap the dense core and get a sample from the error sampler for each bootstrap
        # then we calculate the velocity for each bootstrap and finally report the confidence interval
        # of the differences between the velocities

        # get the dense core
        dense_core_g = dense_sample(self.weights[labels == g])
        dense_core_n = dense_sample(self.weights[labels == n])

        # get the data of the dense core
        data_g = self.data[labels == g].loc[dense_core_g]
        data_n = self.data[labels == n].loc[dense_core_n]

        velocity_differences = []
        for i in range(100):
            # get a bootstrap sample from data_g and data_n
            bootstrap_g = data_g.sample(n=len(data_g) - 1, replace=True)
            bootstrap_n = data_n.sample(n=len(data_n) - 1, replace=True)

            # print(bootstrap_g.shape)
            # print(bootstrap_n.shape)

            # get the error sample for both clusters
            err_sample_mean = []
            for cluster in [bootstrap_g, bootstrap_n]:
                err_sample_mean.append(np.mean(self.get_error_sample(cluster), axis=0))
            
            # calculate the difference
            velocity_differences.append(err_sample_mean[0] - err_sample_mean[1])

        # calculate the confidence interval
        confidence_interval = np.percentile(velocity_differences, [2.5, 97.5], axis=0)

        # check if the confidence interval contains 0 for all dimension
        is_same_velocity = True
        for i in range(3):
            if confidence_interval[0][i] > 0 or confidence_interval[1][0] < 0:
                is_same_velocity = False
                break

        if return_stats:
            return is_same_velocity, velocity_differences, confidence_interval
        return is_same_velocity, velocity_differences
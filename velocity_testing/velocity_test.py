import pandas as pd
import numpy as np
from velocity_testing.calculate_velocity import calculate_velocity
from scipy.stats import ttest_ind
from scipy.spatial import distance
import sys

sys.path.append('/home/nico/VSCodeRepos/SigMA')
from NoiseRemoval.bulk_velocity_solver_matrix import dense_sample, bootstrap_bulk_velocity_solver_matrix
from NoiseRemoval.OptimalVelocity import vr_solver, transform_velocity, optimize_velocity
from NoiseRemoval.xd_special import XDSingleCluster
from miscellaneous.covariance_trafo_sky2gal import transform_covariance_shper2gal
from miscellaneous.error_sampler import ErrorSampler


class VelocityTester:
    def __init__(self, data, weights, testing_mode):
        self.data = data
        self.testing_mode = testing_mode
        self.weights = weights
    
    def run_test(self, labels, g, n, err_sampler=None, return_pvalues=False):
        if self.testing_mode == 'ttest':
            return self.ttest(labels, g, n, return_pvalues)
        elif self.testing_mode == 'bootstrap_range_test':
            return self.bootstrap_range_test(labels, g, n, return_pvalues)
        elif self.testing_mode == 'xd_mean':
            return self.xd_mean(labels, g, n, err_sampler, return_pvalues)
        elif self.testing_mode == 'error_sample_ttest':
            return self.error_sample_ttest(labels, g, n, return_pvalues)
        elif self.testing_mode == 'xd_mean_sample_distance':
            return self.xd_mean_sample_distance(labels, g, n, err_sampler, return_pvalues)


    def ttest(self, labels, g, n, return_pvalues=False):
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

        if return_pvalues:
            return self.is_same_velocity(pvalues), velocity_results, pvalues
        return self.is_same_velocity(pvalues), velocity_results
    
    def bootstrap_range_test(self, labels, g, n, return_pvalues=False):
        # calculate the velocities for both groups
        velocity_results = []
        for cluster in [g, n]:
            velocity_results.append(
                bootstrap_bulk_velocity_solver_matrix(
                    self.data[labels == cluster], 
                    self.weights[labels == cluster],
                    n_bootstraps=5
                )
            )

        # TODO: refactor
        for i in range(2):
            for j in range(len(velocity_results[i])):
                velocity_results[i][j] = velocity_results[i][j].x[:3]
        velocity_results = np.array(velocity_results)


        # calculate the differences between the velocities and confidence intervals of differences
        velocity_differences = velocity_results[0] - velocity_results[1]
        confidence_interval = np.percentile(velocity_differences, [2.5, 97.5], axis=0)


        # check if the confidence interval contains 0 for all dimension
        is_same_velocity = True
        for i in range(3):
            if confidence_interval[0][i] > 0 or confidence_interval[1][0] < 0:
                is_same_velocity = False
                break

        if return_pvalues:
            return is_same_velocity, velocity_results, confidence_interval
        return is_same_velocity, velocity_results
    
    @staticmethod
    def is_same_velocity(pvalues):
        return np.all(pvalues > 0.05)
    
    def get_xd(self, err_sampler, cluster_index):
        # Written by Sebastian Ratzenböck
        # get dense sample
        dense_core = dense_sample(self.weights[cluster_index])

        c_vel = ['U', 'V', 'W']
        X = self.data[cluster_index].loc[dense_core, c_vel].values
        C = err_sampler.C[cluster_index][dense_core, 3:, 3:]

        ra, dec, plx = self.data[cluster_index].loc[dense_core, ['ra', 'dec', 'parallax']].values.T
        # Compute covariance matrix in Galactic coordinates
        C_uvw = transform_covariance_shper2gal(ra, dec, plx, C)     
        xd = XDSingleCluster(max_iter=200, tol=1e-3).fit(X, C_uvw)
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
    
    def xd_mean(self, labels, g, n, err_sampler, return_pvalues=False):
        # for each cluster, get the xd object
        xd = []
        for cluster in [g, n]:
            cluster_index = labels == cluster
            xd.append(self.get_xd(err_sampler, cluster_index))

        # get the mahalanobis distance between the two clusters and maximize it
        max_mahalanobis_distance = max(
            self.calculate_distance(xd[0].V, xd[0].mu, xd[1].mu),
            self.calculate_distance(xd[1].V, xd[1].mu, xd[0].mu)
        )

        return max_mahalanobis_distance < 2, [xd[0].mu, xd[1].mu], max_mahalanobis_distance
    
        
    def get_error_sample(self, cluster_index):
        # generate a new sampler on the cluster data
        err_sampler = ErrorSampler(self.data[cluster_index])
        err_sampler.build_covariance_matrix()
        # Create sample from errors
        cols = ['X', 'Y', 'Z', 'U', 'V', 'W']
        data_new = pd.DataFrame(err_sampler.spher2cart(err_sampler.new_sample()), columns=cols)

        return data_new[['U', 'V', 'W']]
    
    def error_sample_ttest(self, labels, g, n, return_pvalues=False):
        # get the error sample for both clusters
        err_sample = []
        for cluster in [g, n]:
            cluster_index = labels == cluster
            err_sample.append(self.get_error_sample(cluster_index))

        # calculate a t-test on the error samples
        _, pvalues = ttest_ind(err_sample[0], err_sample[1], equal_var=False)

        if return_pvalues:
            return self.is_same_velocity(pvalues), err_sample, pvalues
        return self.is_same_velocity(pvalues), err_sample
    
    def xd_mean_sample_distance(self, labels, g, n, err_sampler, return_pvalues=False):
        # we need the error sampler for both clusters to generate new samples
        # we need the extreme deconvolution to calculate the mahalanobis distance
        # we then calculate the maximal distance from each cluster to the mean of the other cluster
        # and finally take the minimum of those two distances

        samples, xd = [], []
        for cluster in [g, n]:
            cluster_index = labels == cluster
            samples.append(self.get_error_sample(cluster_index))
            xd.append(self.get_xd(err_sampler, cluster_index))

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
import numpy as np
from scipy.stats import ttest_ind
from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components
import sys

sys.path.append('/home/nico/VSCodeRepos/SigMA')
from NoiseRemoval.bulk_velocity_solver_matrix import bootstrap_bulk_velocity_solver_matrix, bulk_velocity_solver_matrix
from NoiseRemoval.xd_outlier import XDOutlier
from miscellaneous.covariance_trafo_sky2gal import transform_covariance_shper2gal
from NoiseRemoval.RemoveNoiseTransformed import remove_noise_simple
from NoiseRemoval.BulkVelocityClassic import ClassicBV


class VelocityTester:
    def __init__(self, data, weights, testing_mode, clusterer, err_sampler=None):
        self.data = data
        self.testing_mode = testing_mode
        self.weights = weights
        self.error_sampler = err_sampler
        self.clusterer = clusterer
        self.cbve = ClassicBV(method='BFGS')
        self.bootstrap_cache = {}
    
    def run_test(self, labels, old_cluster, new_cluster, clusterer=None, return_stats=False):
        if self.testing_mode   == 'ttest':
            return self.ttest                                   (labels, old_cluster, new_cluster, clusterer, return_stats)
        elif self.testing_mode == 'bootstrap_difference_test':
            return self.bootstrap_difference_test               (labels, old_cluster, new_cluster, clusterer, return_stats)
        elif self.testing_mode == 'bootstrap_range_test':
            return self.bootstrap_range_test                    (labels, old_cluster, new_cluster, clusterer, return_stats)
        elif self.testing_mode == 'xd_mean_distance':
            return self.mahalanobis_mean_distance               (labels, old_cluster, new_cluster, clusterer, return_stats, method='xd')
        elif self.testing_mode == 'xd_sample_ttest':
            return self.normal_sample_ttest                     (labels, old_cluster, new_cluster, clusterer, return_stats, method='xd')
        elif self.testing_mode == 'xd_mean_distance_sample_distance':
            return self.minimax_mean_distance_sample_distance    (labels, old_cluster, new_cluster, clusterer, return_stats, method='xd')
        elif self.testing_mode == 'xd_sample_bootstrap_range_test':
            return self.normal_sample_bootstrap_range_test      (labels, old_cluster, new_cluster, clusterer, return_stats, method='xd')
        elif self.testing_mode == 'cbve_mean_distance':
            return self.mahalanobis_mean_distance               (labels, old_cluster, new_cluster, clusterer, return_stats, method='cbve')
        elif self.testing_mode == 'cbve_sample_ttest':
            return self.normal_sample_ttest                     (labels, old_cluster, new_cluster, clusterer, return_stats, method='cbve')
        elif self.testing_mode == 'cbve_mean_distance_sample_distance':
            return self.minimax_mean_distance_sample_distance   (labels, old_cluster, new_cluster, clusterer, return_stats, method='cbve')
        elif self.testing_mode == 'cbve_sample_bootstrap_range_test':
            return self.normal_sample_bootstrap_range_test      (labels, old_cluster, new_cluster, clusterer, return_stats, method='cbve')
        else:
            raise ValueError("Invalid testing mode")
        
    def update_testing_mode(self, testing_mode):
        self.testing_mode = testing_mode
        
    def get_bootstrap_velocities(self, labels, cluster, n_bootstraps=100):
        """
        Get the bootstrap velocities for a cluster

        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        cluster : int
            The label of the cluster
        n_bootstraps : int
            The number of bootstraps

        Returns
        -------
        np.array
            The velocities of the cluster for each bootstrap
        """
        # the cache helps speed up the calculation
        cache_key = labels[labels == cluster].tostring()
        if cache_key in self.bootstrap_cache:
            print('Cluster found in Cache')
            return self.bootstrap_cache[cache_key]

        sol = bootstrap_bulk_velocity_solver_matrix(
            self.data[labels == cluster], 
            self.weights[labels == cluster],
            n_bootstraps=n_bootstraps
        )
        cluster_velocity = np.array([s.x[:3] for s in sol])
        self.bootstrap_cache[cache_key] = cluster_velocity
        return cluster_velocity


    def ttest(self, labels, old_cluster, new_cluster, clusterer, return_stats=False):
        """
        Perform a t-test on the velocities of two clusters

        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        old_cluster : int
            Label of the first cluster
        new_cluster : int
            Label of the second cluster
        return_stats : bool
            Return the statistics of the test

        Returns
        -------
        bool
            True if the velocities are the same, False otherwise
        """
        # calculate the velocities for both clusters
        n_bootstraps = 100
        velocity_results = []
        for cluster in [old_cluster, new_cluster]:
            velocity_results.append(self.get_bootstrap_velocities(labels, cluster, n_bootstraps))

        # calculate the t-statistic and p-value
        t_stat, pvalues = ttest_ind(*velocity_results, equal_var=False)

        # calculate the mean deviation for each cluster
        mean_deviation = [np.mean(np.std(velocity_results[i], axis=0)) for i in range(2)]

        if return_stats:
            return self.is_same_velocity(pvalues), mean_deviation, {'velocity_results': velocity_results, 'pvalues': pvalues, 'stats':t_stat}
        return self.is_same_velocity(pvalues), mean_deviation
    
    def bootstrap_difference_test(self, labels, old_cluster, new_cluster, clusterer, return_stats=False):
        """
        Perform a two-sided bootstrap test.
        
        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        old_cluster : int
            Label of the first cluster
        new_cluster : int
            Label of the second cluster
        return_stats : bool
        
        Returns
        -------
        bool
            True if the velocities are the same, False otherwise"""

        # calculate the true velocities difference
        velocity_true = []
        for cluster in [old_cluster, new_cluster]:
            sol = bulk_velocity_solver_matrix(
                self.data[labels == cluster], 
                self.weights[labels == cluster]
            )
            velocity_true.append(sol.x[:3])
        velocity_differences = velocity_true[0] - velocity_true[1]

        # calculate the velocities for both groups
        n_bootstraps = 100
        velocity_results = []
        for cluster in [old_cluster, new_cluster]:
            velocity_results.append(self.get_bootstrap_velocities(labels, cluster, n_bootstraps))

        # calculate the difference between each bootstrap
        velocity_differences_bootstrap = [
            velocity_results[0][i] - velocity_results[1][i]
            for i in range(n_bootstraps)
        ]

        pvalue = np.max([
            ((velocity_differences_bootstrap > velocity_differences) + 1) / (n_bootstraps + 1), 
            ((velocity_differences_bootstrap < velocity_differences) + 1) / (n_bootstraps + 1)
        ])
        pvalue *= 2

        # calculate the mean deviation for new_cluster
        mean_deviation = [np.mean(np.std(velocity_results[i], axis=0)) for i in range(2)]

        # the pvalue has to be smaller than 0.05 to reject the null hypothesis
        if return_stats:
            return pvalue < 0.05, mean_deviation, {'velocity_difference': velocity_differences_bootstrap, 'pvalue': pvalue}
        return pvalue < 0.05, mean_deviation
    
    def bootstrap_range_test(self, labels, old_cluster, new_cluster, clusterer, return_stats=False):
        """
        Perform a range-based bootstrap test.
        
        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        old_cluster : int
            Label of the first cluster
        new_cluster : int
            Label of the second cluster
        return_stats : bool
        
        Returns
        -------
        bool
            True if the velocities are the same, False otherwise"""

        # calculate the velocities for both groups
        n_bootstraps = 100
        velocity_results = []
        for cluster in [old_cluster, new_cluster]:
            velocity_results.append(self.get_bootstrap_velocities(labels, cluster, n_bootstraps))

        # calculate the difference between each bootstrap
        velocity_differences_bootstrap = [
            velocity_results[0][i] - velocity_results[1][i]
            for i in range(50)
        ]

        # calculate the confidence interval
        confidence_interval = np.percentile(velocity_differences_bootstrap, [2.5, 97.5], axis=0)

        # calculate the mean deviation for new_cluster
        mean_deviation = [np.mean(np.std(velocity_results[i], axis=0)) for i in range(2)]

        # check if the confidence interval contains 0 for all dimension
        is_same_velocity = True
        for i in range(3):
            if confidence_interval[0][i] > 0 or confidence_interval[1][i] < 0:
                is_same_velocity = False
                break

        if return_stats:
            return is_same_velocity, mean_deviation, {'velocity_bootstrap': velocity_differences_bootstrap, 'confidence_interval': confidence_interval}
        return is_same_velocity, mean_deviation
    
    @staticmethod
    def is_same_velocity(pvalues):
        return np.all(pvalues > 0.05)
    
    def extract_cluster_single(self, label_bool_arr, clusterer):
        # Written by Sebastian Ratzenböck
        cluster_bool_array = remove_noise_simple(label_bool_arr, te_obj=clusterer)
        if cluster_bool_array is not None:
            return cluster_bool_array
        else:
            data_idx = np.arange(clusterer.X.shape[0])
            rho = clusterer.weights_[label_bool_arr]
            mad = np.median(np.abs(rho - np.median(rho)))
            threshold = np.median(rho)*0.995 + 3 * mad * 1.05
            # Statistisch fundierterer cut
            # threshold = np.median(rho) + 3 * mad
            idx_cluster = data_idx[label_bool_arr][rho > threshold]
            if len(idx_cluster) > 20:
                # labels_with_noise[idx_cluster] = i
                # Only graph connected points allowed
                _, cc_idx = connected_components(clusterer.A[idx_cluster, :][:, idx_cluster])
                # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
                idx_cluster = data_idx[idx_cluster][cc_idx == np.argmax(np.bincount(cc_idx))]
            
            cluster_bool_array = np.isin(data_idx, idx_cluster)
            return cluster_bool_array
        
    def get_std_from_covariance(self, V):
        # Written by Sebastian Ratzenböck
        eigvals = np.linalg.eigvals(V)
        return np.sqrt(np.mean(eigvals))
    
    def get_xd(self, dense_core):
        # Written by Sebastian Ratzenböck
        X = self.data.loc[dense_core, ['U', 'V', 'W']].values
        C_i = self.error_sampler.C[dense_core, 3:, 3:]
        ra, dec, plx = self.data.loc[dense_core, ['ra', 'dec', 'parallax']].values.T
        Xerr = transform_covariance_shper2gal(ra, dec, plx, C_i)    

        xd = XDOutlier().fit(X, Xerr)
        return xd.min_entropy_component()
    
    def get_classical_bulk_estimate(self, dense_core):
        return self.cbve.estimate_normal_params(dense_core, method='BFGS')
    
    def get_cov_mean(self, cluster_index, clusterer, method='xd'):
        dense_core = self.extract_cluster_single(cluster_index, clusterer)
        # check if dense core is empty
        if np.all(dense_core == False):
            raise ValueError("Cluster too small")

        if method == 'xd':
            return self.get_xd                      (dense_core)
        elif method == 'cbve':
            return self.get_classical_bulk_estimate (dense_core)
        else:
            raise ValueError("Invalid method")

    
    @staticmethod
    def calculate_distance(V, mu, x):
        try:
            # reshape and invert the cov matrix
            iv = np.linalg.inv(V.reshape(3, 3))
        except:
            raise ValueError("Covariance Matrix can not be inverted")

        # flatten the input arrays
        mu = mu.flatten()
        x = x.flatten()

        return distance.mahalanobis(x, mu, iv)
    
    def mahalanobis_mean_distance(self, labels, old_cluster, new_cluster, clusterer, return_stats=False, method='xd'):
        """
        Perform a test based on the mahalanobis distance between the means of two clusters.
        
        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        old_cluster : int
            Label of the first cluster 
        new_cluster : int
            Label of the second cluster
        return_stats : bool
            Return the statistics of the test
                
        Returns
        -------
        bool
            True if the velocities are the same, False otherwise
        """

        # for each cluster, get the xd object
        try:
            sample_params = [
                self.get_cov_mean(labels == cluster, clusterer, method=method)
                for cluster in [old_cluster, new_cluster]
            ]
        except:
            return False, [1000, 1000], {'error': 'Cluster too small'}

        # get the mahalanobis distance between the two clusters and maximize it
        try:
            max_mahalanobis_distance = max([
                VelocityTester.calculate_distance(sample_params[0][1], sample_params[0][0], sample_params[1][0]),
                VelocityTester.calculate_distance(sample_params[1][1], sample_params[1][0], sample_params[0][0])
            ])
        except:
            return False, [1000, 1000], {'error': 'Covariance Matrix is singular'}

        # calculate the mean deviation for new_cluster
        mean_deviation = [self.get_std_from_covariance(sample_params[i][1]) for i in range(2)]

        if return_stats:
            return max_mahalanobis_distance < 2, mean_deviation, {'xd': [sample_params[0], sample_params[1]], 'mahalanobis_distance': max_mahalanobis_distance}
        return max_mahalanobis_distance < 2, mean_deviation
        
    def create_cluster_sample(self, cluster_index, clusterer, method='xd'):
        sample_params = self.get_cov_mean(cluster_index, clusterer, method=method)
        # create samples from a normal distribution using cov and mean
        return np.random.multivariate_normal(sample_params[0], sample_params[1], 500)
    
    def normal_sample_ttest(self, labels, old_cluster, new_cluster, clusterer, return_stats=False, method='xd'):
        """
        Perform a t-test on the error samples of two clusters.
        
        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        old_cluster: int
            Label of the first cluster
        new_cluster: int
            Label of the second cluster
        return_stats : bool
            Return the statistics of the test
            
        Returns
        -------
        bool
            True if the velocities are the same, False otherwise"""

        # get a cluster sample for both clusters
        try:
            cluster_samples = [
                self.create_cluster_sample(labels == cluster, clusterer, method=method)
                for cluster in [old_cluster, new_cluster]
            ]
        except:
            return False, [1000, 1000], {'error': 'Cluster too small'}

        # calculate a t-test on the error samples
        t_stat, pvalues = ttest_ind(*cluster_samples, equal_var=False)

        # calculate the mean deviation for both clusters
        mean_deviation = [np.mean(np.std(cluster_samples[i], axis=0)) for i in range(2)]

        if return_stats:
            return self.is_same_velocity(pvalues), mean_deviation, {'err_samples': cluster_samples, 'p_values': pvalues, 't_stat': t_stat}
        return self.is_same_velocity(pvalues), mean_deviation
    
    def minimax_mean_distance_sample_distance(self, labels, old_cluster, new_cluster, clusterer, return_stats=False, method='xd'):
        """
        Perform a test based on the mahalanobis distance between the means of two clusters.

        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        old_cluster : int
            Label of the first cluster
        new_cluster : int
            Label of the second cluster
        return_stats : bool
            Return the statistics of the test

        Returns
        -------
        bool
            True if the velocities are the same, False otherwise
        """

        try:
            sample_params = [
                self.get_cov_mean(labels == cluster, clusterer, method=method)
                for cluster in [old_cluster, new_cluster]
            ]
        except:
            return False, [1000, 1000], {'error': 'Cluster too small'}

        # calculate the maximal distance between each point in the samples and the mean of the other cluster
        distances = []
        j = 1
        for cluster in [old_cluster, new_cluster]:
            cluster_index = labels == cluster
            # calculate the mahalanobis distance between all points in one cluster and the mean of the other cluster
            try:
                distances.append(
                    max(
                        [
                            self.calculate_distance(sample_params[j][1], sample_params[j][0], x)
                            for x in self.data[cluster_index].loc[:, ['U', 'V', 'W']].values
                        ]
                    )
                )
            except:
                return False, [1000, 1000], {'error': 'Covariance Matrix is singular'}
            j -= 1

        # take the minimum of the two distances
        min_distance = min(distances)

        # calculate the mean deviation for both clusters
        mean_deviation = [self.get_std_from_covariance(sample_params[i][1]) for i in range(2)]

        if return_stats:
            return min_distance < 2, mean_deviation, {'xd': [sample_params[0], sample_params[1]], 'distance': min_distance}
        return min_distance < 2, mean_deviation
    
    def normal_sample_bootstrap_range_test(self, labels, old_cluster, new_cluster, clusterer, return_stats, method='xd'):
        """
        Perform a range-based bootstrap test on the samples taken from the xd.

        Parameters
        ----------
        labels : np.array
            The labels of the SigMA clustering
        g : int
            Label of the first cluster
        n : int
            Label of the second cluster
        return_stats : bool
            Return the statistics of the test

        Returns
        -------
        bool
            True if the velocities are the same, False otherwise
        """

        # get samples
        try:
            velocity_samples = [
                self.create_cluster_sample(labels == cluster, clusterer, method=method)
                for cluster in [old_cluster, new_cluster]
            ]
        except:
            return False, [1000, 1000], {'error': 'Cluster too small'}

        velocity_differences = []
        for i in range(100):
            sample_mean = []
            for sample in velocity_samples:
                bootstrapped_sample = sample[np.random.choice(len(sample), len(sample) - 1, replace=True)]
                sample_mean.append(np.mean(bootstrapped_sample, axis=0))

            velocity_differences.append(sample_mean[0] - sample_mean[1])

        # calculate the confidence interval
        confidence_interval = np.percentile(velocity_differences, [2.5, 97.5], axis=0)

        # check if the confidence interval contains 0 for all dimension
        is_same_velocity = True
        for i in range(3):
            if confidence_interval[0][i] > 0 or confidence_interval[1][i] < 0:
                is_same_velocity = False
                break

        # calculate the mean deviation for new_cluster
        mean_deviation = [np.mean(np.std(self.data[labels == cluster].loc[:, ['U', 'V', 'W']], axis=0)) for cluster in [old_cluster, new_cluster]]

        if return_stats:
            return is_same_velocity, mean_deviation, {'velocity_difference': velocity_differences, 'confidence_interval': confidence_interval}
        return is_same_velocity, mean_deviation
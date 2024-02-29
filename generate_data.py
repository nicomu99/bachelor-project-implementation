import pandas as pd
import numpy as np
import sys

sys.path.append('/home/nico/VSCodeRepos/SigMA/')
from miscellaneous.error_sampler import ErrorSampler

class DataGenerator:
    # The contents of this class were mainly written by Sebastian Ratzenböck
    def __init__(self, fname='/home/nico/VSCodeRepos/bachelor-project-implementation/simulated_data/data_orion_focus.csv'):
        self.data_gaia = pd.read_csv(fname)
        self.df = None
        self.df_noisy = None
        self.labels = None
        self.labels_noisy = None
        self.rv_isnan = None

    def generate_data(self, xyz_mean, uvw_mean, xyz_std, uvw_std, n_samples):
        """Inputs are lists of numpy arrays or lists of integers in the case of n_samples. The lists must have the same length."""
        data_simulated = []
        labels_list = []
        # generate data for each cluster
        cluster_id = 0
        for Xmu_i, Umu_i, Xstd_i, Ustd_i, n_i in zip(xyz_mean, uvw_mean, xyz_std, uvw_std, n_samples):
            X = np.concatenate((Xmu_i, Umu_i))
            std = np.concatenate((Xstd_i, Ustd_i))
            N = max(n_i, 200)
            C = np.diag(std**2)
            cluster_simulated = np.random.multivariate_normal(X, C, N)
            data_simulated.append(cluster_simulated)
            labels_list.append(np.ones(N, dtype=int)*cluster_id)
            cluster_id += 1
        
        data_simulated = np.vstack(data_simulated) # stack all data
        self.labels = np.hstack(labels_list) # stack all labels

        # Simulate a Gaussian cluster in 6D
        cols = ['X', 'Y', 'Z', 'U', 'V', 'W']
        self.df = pd.DataFrame(data_simulated, columns=cols)

        # Sample from the ErrorSampler
        ra, dec, plx, pmra, pmdec, rv = ErrorSampler().cart2spher(self.df[cols].values)
        self.df['ra'] = ra
        self.df['dec'] = dec
        self.df['parallax'] = plx
        self.df['pmra'] = pmra
        self.df['pmdec'] = pmdec
        self.df['radial_velocity'] = rv

        return self.df, self.labels
        
    def add_noise(self, n_samples=50_000, delta_perc=20):
        cols = ['X', 'Y', 'Z', 'U', 'V', 'W']
        cols2match = [
            'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
            'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
            'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr'
        ]
        # Add bg noise
        ptp_data = self.df[cols].max() - self.df[cols].min()
        ranges = [
            (self.df[col].min() - ptp_data[col] * delta_perc, self.df[col].max() + ptp_data[col] * delta_perc) 
                for col in cols
        ]
        # Create uniform noise
        noise = pd.DataFrame(np.random.uniform(*zip(*ranges), (n_samples, len(cols))), columns=cols)
        ra, dec, plx, pmra, pmdec, rv = ErrorSampler().cart2spher(noise[cols].values)
        noise['ra'] = ra
        noise['dec'] = dec
        noise['parallax'] = plx
        noise['pmra'] = pmra
        noise['pmdec'] = pmdec
        noise['radial_velocity'] = rv
        cols_shere = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
        # Add noise to data
        self.df_noisy = pd.concat([self.df[cols + cols_shere], noise[cols + cols_shere]], axis=0)
        self.labels_noisy = np.r_[self.labels, np.ones(n_samples) * -1].astype(int)

        # Keep only data within cluster range
        self.df_noisy[cols2match] = self.data_gaia[cols2match].sample(n=self.df_noisy.shape[0], replace=True).values
        self.rv_isnan = self.df_noisy['radial_velocity_error'].isna().values.ravel()
        self.df_noisy.loc[self.rv_isnan, 'radial_velocity_error'] = 1e3
        self.df_noisy.loc[self.rv_isnan, 'radial_velocity'] = 0.0

        err_sampler = ErrorSampler(self.df_noisy)
        err_sampler.build_covariance_matrix()
        # Create sample from errors
        data_new = pd.DataFrame(err_sampler.new_sample(), columns=cols_shere)
        data_new_cart = pd.DataFrame(err_sampler.spher2cart(data_new.values), columns=cols)
        self.df_noisy = pd.concat([data_new, data_new_cart, self.df_noisy[cols2match].reset_index(drop=True)], axis=1)

        # Set radial velocity values to NaNs --> only where the original data had NaNs!!
        self.df_noisy.loc[self.rv_isnan, 'radial_velocity_error'] = np.nan
        self.df_noisy.loc[self.rv_isnan, 'radial_velocity'] = np.nan

        return self.df_noisy, self.labels_noisy, err_sampler
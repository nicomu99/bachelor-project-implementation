import pandas as pd
import numpy as np
import sys

sys.path.append('/home/nico/VSCodeRepos/SigMA/')
from miscellaneous.error_sampler import ErrorSampler

class DataGenerator:
    # The contents of this class were mainly written by Sebastian Ratzenböck
    def __init__(self):
        self.data_gaia = pd.read_csv('simulated_data/data_orion_focus.csv')
        self.df_xyz = pd.read_csv('simulated_data/Simulated_clusters_labeled_Region0_run6.csv')
        self.df_uvw = pd.read_csv('simulated_data/UVW_stats_region0_new.csv')

        self.X_means = self.df_xyz.groupby('label').mean()[['X', 'Y', 'Z']]
        self.X_means['X_std'] = [3, 4, 5, 6, 7, 8]
        self.X_means['Y_std'] = [3, 4, 5, 6, 7, 8]
        self.X_means['Z_std'] = [3, 4, 5, 6, 7, 8]

        new_idx = self.df_uvw.columns[1:]
        new_cols = self.df_uvw['Cluster_id new'].values
        self.df_uvw = pd.DataFrame(self.df_uvw[new_idx].T.values, columns=new_cols, index=new_idx).rename(
            columns={'mean U': 'U', 'mean V': 'V', 'mean W': 'W', 
                'SD U': 'U_std', 'SD V': 'V_std', 'SD W': 'W_std'},
        )

        # for the first 6 clusters, add uvw info
        self.df_infos = pd.concat((self.X_means, self.df_uvw.iloc[:6].reset_index(inplace=False, drop=True)), axis=1)
        self.df_infos['U_std'] = 2
        self.df_infos['V_std'] = 2
        self.df_infos['W_std'] = 2

        self.df = None
        self.labels = None

    def generate_data(self, test_cases):
        mu_cols_velocity = ['U', 'V', 'W']
        std_cols = ['U_std', 'V_std', 'W_std']
        data_simulated = []
        self.labels = []

        # generate data for each cluster
        cluster_id = 0
        for test_case in test_cases:
            for labels, mu_position, mu_std in zip(test_case['clusters'], test_case['mu_position'], test_case['mu_std']):
                mu_velocity = self.df_infos[mu_cols_velocity].iloc[labels].values
                std = np.concatenate((mu_std, self.df_infos[std_cols].iloc[labels].values))
                N = max(int(self.df_infos[['n_cluster']].iloc[labels].values), 200)
                C = np.diag(std**2)
                cluster_simulated = np.random.multivariate_normal(np.concatenate((mu_position, mu_velocity)), C, N)
                data_simulated.append(cluster_simulated)
                self.labels.append(np.ones(N, dtype=int)*cluster_id)
                cluster_id += 1
            
        data_simulated = np.vstack(data_simulated) # stack all data
        self.labels = np.hstack(self.labels) # stack all labels

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
        
    def add_noise(self, n_samples=50_000):
        cols = ['X', 'Y', 'Z', 'U', 'V', 'W']
        cols2match = [
            'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
            'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
            'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr'
        ]
        # Add bg noise
        delta_perc = 50
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
        self.df = pd.concat([self.df[cols + cols_shere], noise[cols + cols_shere]], axis=0)
        self.labels = np.r_[self.labels, np.ones(n_samples) * -1].astype(int)

        # Compute distances to cluster centers
        dists = np.sqrt(np.sum(self.X_means.values**2, axis=1))
        dx = 2
        dmin, dmax = dists.min()-dx, dists.max()+dx 
        self.data_gaia['dist'] = np.sqrt(np.sum(self.data_gaia[['X', 'Y', 'Z']].values**2, axis=1))
        self.data_gaia = self.data_gaia.loc[(self.data_gaia['dist'] > dmin) & (self.data_gaia['dist'] < dmax)]

        # Keep only data within cluster range
        self.df[cols2match] = self.data_gaia[cols2match].sample(n=self.df.shape[0], replace=True).values
        self.df.loc[self.df['radial_velocity_error'].isna().values.ravel(), 'radial_velocity_error'] = 1e3

        err_sampler = ErrorSampler(self.df)
        err_sampler.build_covariance_matrix()
        # Create sample from errors
        data_new = pd.DataFrame(err_sampler.new_sample(), columns=cols_shere)
        data_new_cart = pd.DataFrame(err_sampler.spher2cart(data_new.values), columns=cols)
        data = pd.concat([data_new, data_new_cart, self.df[cols2match].reset_index(drop=True)], axis=1)

        return data, self.labels, err_sampler
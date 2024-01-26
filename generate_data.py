import pandas as pd
import numpy as np
import sys

sys.path.append('/home/nico/VSCodeRepos/SigMA/')
from miscellaneous.error_sampler import ErrorSampler


def generate_data():
    # read files
    data_gaia = pd.read_csv('simulated_data/data_orion_focus.csv')
    fname_xyz = 'simulated_data/Simulated_clusters_labeled_Region0_run6.csv'
    fname_uvw = 'simulated_data/UVW_stats_region0_new.csv'

    # read file with xyz info, file contains about 9000 stars with all information
    df_xyz = pd.read_csv(fname_xyz)
    # calculate means and generate std for each cluster
    X_means = df_xyz.groupby('label').mean()[['X', 'Y', 'Z']]
    X_means['X_std'] = np.random.uniform(3, 8, size=len(X_means))
    X_means['Y_std'] = np.random.uniform(3, 8, size=len(X_means))
    X_means['Z_std'] = np.random.uniform(3, 8, size=len(X_means))

    # read file with uvw info
    # contains mean U, V, W and std U, V, W for each cluster and value counts
    df_uvw = pd.read_csv(fname_uvw)
    uvw_idx_names = ['mean U', 'mean V', 'mean W'] #, 'std U', 'std V', 'std W']
    new_idx = df_uvw.columns[1:]
    new_cols = df_uvw['Cluster_id new'].values
    df_uvw = pd.DataFrame(df_uvw[new_idx].T.values, columns=new_cols, index=new_idx).rename(
        columns={'mean U': 'U', 'mean V': 'V', 'mean W': 'W', 
                'SD U': 'U_std', 'SD V': 'V_std', 'SD W': 'W_std'},
    )

    # for the first 6 clusters, add uvw info
    df_infos = pd.concat((X_means, df_uvw.iloc[:6].reset_index(inplace=False, drop=True)), axis=1)
    df_infos['U_std'] = 2
    df_infos['V_std'] = 2
    df_infos['W_std'] = 2

    mu_cols = ['X', 'Y', 'Z', 'U', 'V', 'W']
    std_cols = ['X_std', 'Y_std', 'Z_std', 'U_std', 'V_std', 'W_std']
    data_simulated = []
    labels = []
    for i in range(df_infos.shape[0]):
        mu = df_infos[mu_cols].iloc[i].values # get the means for each cluster
        std = df_infos[std_cols].iloc[i].values # get the std for each cluster
        N = max(int(df_infos[['n_cluster']].iloc[i].values), 50) # either 50 or n_cluster
        C = np.diag(std**2) # covariance matrix with variance on diagonal
        data_simulated.append(np.random.multivariate_normal(mu, C, N)) # generate N samples from multivariate normal
        labels.append(np.ones(N, dtype=int)*i) # add labels
    data_simulated = np.vstack(data_simulated) # stack all data
    labels = np.hstack(labels) # stack all labels

    # Simulate a Gaussian cluster in 6D
    cols = ['X', 'Y', 'Z', 'U', 'V', 'W']
    df = pd.DataFrame(data_simulated, columns=cols)

    # Sample from the ErrorSampler
    cols2match = [
        'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
        'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr'
    ]
    ra, dec, plx, pmra, pmdec, rv = ErrorSampler().cart2spher(df[cols].values)
    df['ra'] = ra
    df['dec'] = dec
    df['parallax'] = plx
    df['pmra'] = pmra
    df['pmdec'] = pmdec
    df['radial_velocity'] = rv

    # Add bg noise
    n_samples=50_000
    delta_perc = 50
    ptp_data = df[cols].max() - df[cols].min()
    ranges = [
        (df[col].min() - ptp_data[col] * delta_perc, df[col].max() + ptp_data[col] * delta_perc) 
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
    df = pd.concat([df[cols + cols_shere], noise[cols + cols_shere]], axis=0)
    labels_true = np.r_[labels, np.ones(n_samples) * -1].astype(int)

    # Compute distances to cluster centers
    dists = np.sqrt(np.sum(X_means.values**2, axis=1))
    dx = 2
    dmin, dmax = dists.min()-dx, dists.max()+dx 
    data_gaia['dist'] = np.sqrt(np.sum(data_gaia[['X', 'Y', 'Z']].values**2, axis=1))
    data_gaia = data_gaia.loc[(data_gaia['dist'] > dmin) & (data_gaia['dist'] < dmax)]

    # Keep only data within cluster range
    df[cols2match] = data_gaia[cols2match].sample(n=df.shape[0], replace=True).values
    df.loc[df['radial_velocity_error'].isna().values.ravel(), 'radial_velocity_error'] = 1e3

    err_sampler = ErrorSampler(df)
    err_sampler.build_covariance_matrix()
    # Create sample from errors
    data_new = pd.DataFrame(err_sampler.spher2cart(err_sampler.new_sample()), columns=cols)

    return data_new, labels_true, df
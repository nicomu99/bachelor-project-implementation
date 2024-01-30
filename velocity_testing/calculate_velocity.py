import numpy as np
import sys

sys.path.append('/home/nico/VSCodeRepos/SigMA')
from NoiseRemoval.bulk_velocity_solver import dense_sample, optimize_velocity_skycoords
from NoiseRemoval.OptimalVelocity import vr_solver, transform_velocity, optimize_velocity

# CITATION: This code is taken directly from SigMA
def calculate_velocity(
    data, 
    rho,
    ra_col="ra",
    dec_col="dec",
    plx_col="parallax",
    pmra_col="pmra",
    pmdec_col="pmdec",
    rv_col="radial_velocity",
    pmra_err_col="pmra_error",
    pmdec_err_col="pmdec_error",
    rv_err_col="radial_velocity_error",
    rv_max=100,
    rv_min=-100,
):
    cols = [
        ra_col, 
        dec_col, 
        plx_col, 
        pmra_col, 
        pmdec_col, 
        rv_col, 
        pmra_err_col, 
        pmdec_err_col, 
        rv_err_col
    ] 

    # extract dense core
    cut_dense_core = dense_sample(rho)
    ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err = data.iloc[
        cut_dense_core
    ][cols].values.T

    # Calculate a first guess of the optimal velocity
    rv_copy = np.copy(rv)
    rv_copy[np.isnan(rv)] = 0
    rv_err_copy = np.copy(rv_err)
    rv_err_copy[np.isnan(rv_err)] = 1e5
    mean_uvw = np.zeros(3)
    uvw_cols = ["U", "V", "W"]
    if uvw_cols is not None:
        mean_uvw = np.mean(data.iloc[cut_dense_core][uvw_cols], axis=0)
    sol = optimize_velocity(
        ra, dec, plx, pmra, pmdec, rv_copy, pmra_err, pmdec_err, rv_err_copy, init_guess=mean_uvw
    )
    optimal_vel = sol.x[:3]
    print(optimal_vel)

    rv_computed = np.copy(rv)
    rv_isnan_or_large_err = np.isnan(rv_computed)
    rv_computed[rv_isnan_or_large_err] = vr_solver(
        U=optimal_vel[0],
        V=optimal_vel[1],
        W=optimal_vel[2],
        ra=ra[rv_isnan_or_large_err],
        dec=dec[rv_isnan_or_large_err],
        plx=plx[rv_isnan_or_large_err],
        pmra=pmra[rv_isnan_or_large_err],
        pmdec=pmdec[rv_isnan_or_large_err],
    )

    rv_computed[rv_isnan_or_large_err & (rv_computed > rv_max)] = rv_max
    rv_computed[rv_isnan_or_large_err & (rv_computed < rv_min)] = rv_min

    # Transform to uvw
    uvw_computed = transform_velocity(ra, dec, plx, pmra, pmdec, rv_computed)

    return uvw_computed
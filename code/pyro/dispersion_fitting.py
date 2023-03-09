from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# questions:
# size factor set to 10000?
# missing values for fitting dispersions
# fit trend or no?
# do i need dispersion variance


def get_mean_normalized_count_per_guide(X: pd.DataFrame, size_factor=10_000):
    depth = X.sum(axis=0)
    return ((X + 1) / (depth[None, :].divide(size_factor))).mean(axis=1)


def _fit_dispersion_each(y, design_matrix, fit_aux_ols=True):
    """
    y: guide count observation
    design_matrix: design matrix for each observation with covariates
    """
    if (y == 0).all():
        return np.nan
    # Fit Poisson model
    sm_fit = sm.GLM(
        y,
        design_matrix,
        family=sm.families.NegativeBinomial(alpha=0.01, link=sm.families.links.log()),
        missing="drop",
    ).fit()

    if not fit_aux_ols:
        return sm_fit.mu
    y_aux_ols = ((y - sm_fit.mu) ** 2 - sm_fit.mu) / sm_fit.mu
    x_aux_ols = sm_fit.mu
    df_aux_ols = pd.DataFrame({"y": y_aux_ols, "x": x_aux_ols})
    aux_ols_results = smf.ols("y ~ x - 1", df_aux_ols).fit()
    return aux_ols_results.params[0]


def fit_dispersion_all(X: pd.DataFrame, design_matrix) -> np.ndarray:
    """Fit dispersion for each row (guide)
    adata: n_guide x n_sample
    design_matrix: design matrix compatible with statsmodels.GLM indicating the covariats for each sample in adata columns.
    """

    return np.array(
        [
            _fit_dispersion_each(X.iloc[i, :], design_matrix, fit_aux_ols=True)
            for i in range(X.shape[0])
        ]
    )


def fit_dispersion_trend(alphas: Iterable, guide_mean: pd.Series):
    """
    Fit Gamma GLM with identity link to fit
    alpha ~ 1 + 1/mu
    where mu is the normalized mean counts for each guide.

    Args
    ---
    alphas : List of alphas fitted for each guide.
    guide_mean : Normalized mean counts for each guide.

    """
    alpha_trend_df = pd.DataFrame(
        {"inv_mu": 1 / guide_mean.flatten(), "alphas": alphas}
    )
    return smf.glm(
        formula="alphas ~ 1 + inv_mu",
        data=alpha_trend_df,
        family=sm.families.Gamma(link=sm.families.links.identity()),
    ).fit()


def estimate_dispersion_variance(alphas, fitted_alphas) -> np.ndarray:
    """
    Assuming logNormal distribution of dispersions around fitted fitted dispersion values,
    return the MLE estimate of variance of the dispersions.
    """
    return np.nanmean((np.log(alphas) - np.log(fitted_alphas)) ** 2)


def get_trend_fitted_dispersion(
    X: pd.DataFrame,
    design_matrix: pd.DataFrame,
    fit_trend: bool = True,
    min_thres: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Do something similar to DESeq dispersion fitting.
    1. Fit dispersion for each guide.
    2. Fit trend of dispersion parameters as the function of normalized_mean_counts

    Args
    --
    adata: n_guide x n_sample
    design_matrix: design matrix indicating the covariats for each sample in adata columns. (Consider using patsy.dmatrix to produce one.)
    min_thres: If fit_trend is False, dispersion estimate less than the threshold is substituted for trend-fitted dispersion estimate.
    """
    alphas = fit_dispersion_all(X, design_matrix)
    normalized_mean_counts = get_mean_normalized_count_per_guide(X)
    alpha_trend_fit = fit_dispersion_trend(alphas, normalized_mean_counts)
    fitted_alpha = alpha_trend_fit.predict(exog={"inv_mu": 1 / normalized_mean_counts})
    disp_var_est = estimate_dispersion_variance(alphas, fitted_alpha)

    if fit_trend:
        return (fitted_alpha, disp_var_est)

    # Here, trend-fitted dispersion values are used for guides with invalid dispersion estimates
    invalid_disp_idx = np.where((alphas < min_thres) | np.isnan(alphas))[0]
    alphas[invalid_disp_idx] = fitted_alpha[invalid_disp_idx]
    return (alphas, disp_var_est)

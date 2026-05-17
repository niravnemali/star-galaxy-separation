import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc

import sys
sys.path.append('../src')
from config import model_flux_diff, model_flux_mag


DEFAULT_DM_RANGE = (-0.5, 1.5)
DEFAULT_CMODEL_RANGE = (15.0, 25.0)
DEFAULT_CMODEL_BINS = [
    (15.0, 18.0),
    (18.0, 19.0),
    (19.0, 20.0),
    (20.0, 21.0),
    (21.0, 22.0),
    (22.0, 23.0),
    (23.0, 24.0),
    (24.0, 25.0),
]


def _gauss(x, mu, sigma):
    sigma = np.abs(sigma) + 1e-12
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def _skewnorm(x, mu, sigma, alpha):
    sigma = np.abs(sigma) + 1e-12
    t = (x - mu) / sigma
    return 2.0 * _gauss(x, mu, sigma) * 0.5 * erfc(-alpha * t / np.sqrt(2.0))


def _mixture_pdf_skewnorm(x, params, n_resolved):
    # params layout:
    #   [sigmaU, wR1, muR1, sigmaR1, alphaR1, wR2, muR2, sigmaR2, alphaR2, ...]
    # Unresolved component is a Gaussian centered on 0 with free sigma.
    # Resolved components are skew-normal distributions.
    sigmaU = params[0]
    wR = np.array([params[1 + 4 * k] for k in range(n_resolved)])
    wR = np.clip(wR, 0.0, 1.0)
    wU = max(1.0 - np.sum(wR), 0.0)

    y = wU * _gauss(x, 0.0, sigmaU)
    for k in range(n_resolved):
        mu = params[2 + 4 * k]
        sig = params[3 + 4 * k]
        alpha = params[4 + 4 * k]
        y = y + wR[k] * _skewnorm(x, mu, sig, alpha)
    return y


def _initial_guess_skewnorm(n_resolved):
    mus = [0.2, 0.5, 0.9][:n_resolved]
    sigs = [0.15, 0.25, 0.35][:n_resolved]
    alphas = [3.0, 3.0, 3.0][:n_resolved]
    wR = [0.3 / n_resolved] * n_resolved
    p0 = [0.02]
    for k in range(n_resolved):
        p0 += [wR[k], mus[k], sigs[k], alphas[k]]
    return p0


def _bounds_skewnorm(n_resolved, dm_range):
    lo = [1e-4]
    hi = [1.0]
    for _ in range(n_resolved):
        lo += [0.0, 0.0, 1e-3, -20.0]
        hi += [1.0, dm_range[1], 2.0, 20.0]
    return (lo, hi)


def _param_names_skewnorm(n_resolved):
    names = ['sigmaU']
    for k in range(n_resolved):
        names += [f'wR{k+1}', f'muR{k+1}', f'sigmaR{k+1}', f'alphaR{k+1}']
    return names


# --- free star mean variants ---

def _mixture_pdf_skewnorm_freemu(x, params, n_resolved):
    muU = params[0]
    sigmaU = params[1]
    wR = np.array([params[2 + 4 * k] for k in range(n_resolved)])
    wR = np.clip(wR, 0.0, 1.0)
    wU = max(1.0 - np.sum(wR), 0.0)
    y = wU * _gauss(x, muU, sigmaU)
    for k in range(n_resolved):
        mu = params[3 + 4 * k]
        sig = params[4 + 4 * k]
        alpha = params[5 + 4 * k]
        y = y + wR[k] * _skewnorm(x, mu, sig, alpha)
    return y


def _initial_guess_skewnorm_freemu(n_resolved):
    mus = [0.2, 0.5, 0.9][:n_resolved]
    sigs = [0.15, 0.25, 0.35][:n_resolved]
    alphas = [3.0, 3.0, 3.0][:n_resolved]
    wR = [0.3 / n_resolved] * n_resolved
    p0 = [0.0, 0.02]
    for k in range(n_resolved):
        p0 += [wR[k], mus[k], sigs[k], alphas[k]]
    return p0


def _bounds_skewnorm_freemu(n_resolved, dm_range, mu_range=(-0.05, 0.02)):
    lo = [mu_range[0], 1e-4]
    hi = [mu_range[1], 1.0]
    for _ in range(n_resolved):
        lo += [0.0, 0.0, 1e-3, -20.0]
        hi += [1.0, dm_range[1], 2.0, 20.0]
    return (lo, hi)


def _param_names_skewnorm_freemu(n_resolved):
    names = ['muU', 'sigmaU']
    for k in range(n_resolved):
        names += [f'wR{k+1}', f'muR{k+1}', f'sigmaR{k+1}', f'alphaR{k+1}']
    return names


def _mixture_pdf(x, params, n_resolved):
    # params layout:
    #   [sigmaU, wR1, muR1, sigmaR1, wR2, muR2, sigmaR2, ...]
    # Unresolved component is a Gaussian centered on 0 with free sigma and
    # weight wU = 1 - sum(wRk). Weights are clipped to [0, 1] at eval time.
    sigmaU = params[0]
    wR = np.array([params[1 + 3 * k] for k in range(n_resolved)])
    wR = np.clip(wR, 0.0, 1.0)
    wU = max(1.0 - np.sum(wR), 0.0)

    y = wU * _gauss(x, 0.0, sigmaU)
    for k in range(n_resolved):
        mu = params[2 + 3 * k]
        sig = params[3 + 3 * k]
        y = y + wR[k] * _gauss(x, mu, sig)
    return y


def _initial_guess(n_resolved):
    # sigmaU small (unresolved ~ photometric noise), resolved means at 0.2, 0.5, 0.9
    mus = [0.2, 0.5, 0.9][:n_resolved]
    sigs = [0.15, 0.25, 0.35][:n_resolved]
    # split resolved weight evenly; unresolved gets whatever is left
    wR = [0.3 / n_resolved] * n_resolved
    p0 = [0.02]
    for k in range(n_resolved):
        p0 += [wR[k], mus[k], sigs[k]]
    return p0


def _bounds(n_resolved, dm_range):
    lo = [1e-4]
    hi = [1.0]
    for _ in range(n_resolved):
        lo += [0.0, 0.0, 1e-3]
        hi += [1.0, dm_range[1], 2.0]
    return (lo, hi)


def fit_slices(
    df,
    filter_band,
    n_resolved=2,
    cmodel_bins=None,
    dm_range=DEFAULT_DM_RANGE,
    n_hist_bins=300,
    min_counts=50,
    model_type='gaussian',
    free_star_mean=False,
):
    """Fit unresolved+resolved mixture to PSF-CModel in CModel slices.

    model_type : 'gaussian' — all-Gaussian mixture (default, as in v2).
                 'skewnorm' — Gaussian for unresolved + skew-normal for resolved.
    free_star_mean : if True and model_type=='skewnorm', allow the unresolved
                     Gaussian mean to float (bounded to [-0.05, 0.02]).

    Returns a list of dicts, one per CModel bin, each with keys:
        cmodel_lo, cmodel_hi, cmodel_center, n, params, cov, bin_centers,
        counts_norm, success, n_resolved, model_type, free_star_mean.
    """
    if cmodel_bins is None:
        cmodel_bins = DEFAULT_CMODEL_BINS

    if model_type == 'skewnorm' and free_star_mean:
        pdf_func = _mixture_pdf_skewnorm_freemu
        guess_func = _initial_guess_skewnorm_freemu
        bounds_func = _bounds_skewnorm_freemu
    elif model_type == 'skewnorm':
        pdf_func = _mixture_pdf_skewnorm
        guess_func = _initial_guess_skewnorm
        bounds_func = _bounds_skewnorm
    else:
        pdf_func = lambda x, p, nr: _mixture_pdf(x, p, nr)
        guess_func = _initial_guess
        bounds_func = _bounds

    dm_col = f'{filter_band}{model_flux_diff}'
    cm_col = f'{filter_band}{model_flux_mag}'

    results = []
    for lo, hi in cmodel_bins:
        mask = (
            (df[cm_col] >= lo)
            & (df[cm_col] < hi)
            & np.isfinite(df[dm_col])
            & np.isfinite(df[cm_col])
            & (df[dm_col] > dm_range[0])
            & (df[dm_col] < dm_range[1])
        )
        vals = df.loc[mask, dm_col].to_numpy()
        slice_result = {
            'cmodel_lo': lo,
            'cmodel_hi': hi,
            'cmodel_center': 0.5 * (lo + hi),
            'n': len(vals),
            'n_resolved': n_resolved,
            'model_type': model_type,
            'free_star_mean': free_star_mean,
            'params': None,
            'cov': None,
            'bin_centers': None,
            'counts_norm': None,
            'success': False,
        }

        if len(vals) < min_counts:
            results.append(slice_result)
            continue

        counts, edges = np.histogram(vals, bins=n_hist_bins, range=dm_range, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        bin_width = edges[1] - edges[0]
        raw_counts = counts * len(vals) * bin_width
        fit_sigma = np.sqrt(raw_counts + 1.0) / (len(vals) * bin_width)

        try:
            popt, pcov = curve_fit(
                lambda x, *p: pdf_func(x, p, n_resolved),
                centers,
                counts,
                p0=guess_func(n_resolved),
                bounds=bounds_func(n_resolved, dm_range),
                sigma=fit_sigma,
                absolute_sigma=False,
                maxfev=20000,
            )
            slice_result['params'] = popt
            slice_result['cov'] = pcov
            slice_result['success'] = True
        except Exception as e:
            slice_result['error'] = str(e)

        slice_result['bin_centers'] = centers
        slice_result['counts_norm'] = counts
        results.append(slice_result)

    return results


def plot_slice_fits(
    fit_results,
    filter_band,
    dm_range=DEFAULT_DM_RANGE,
    name=None,
    file_path='../plots/',
):
    """One panel per CModel bin: histogram + fitted mixture + components."""
    n = len(fit_results)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    x_plot = np.linspace(dm_range[0], dm_range[1], 600)

    for i, r in enumerate(fit_results):
        ax = axes[i]
        title = f"{filter_band}: {r['cmodel_lo']:.1f}-{r['cmodel_hi']:.1f}  N={r['n']}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('PSF - CModel')
        ax.set_ylabel('density')
        ax.grid(True, alpha=0.3)

        if r['counts_norm'] is not None:
            ax.step(r['bin_centers'], r['counts_norm'], where='mid', color='k', lw=0.8, label='data')

        if r['success']:
            p = r['params']
            nR = r['n_resolved']
            mt = r.get('model_type', 'gaussian')
            fsm = r.get('free_star_mean', False)

            if mt == 'skewnorm' and fsm:
                ax.plot(x_plot, _mixture_pdf_skewnorm_freemu(x_plot, p, nR),
                        color='red', lw=1.2, label='model')
                muU = p[0]
                sigmaU = p[1]
                wR = np.array([p[2 + 4 * k] for k in range(nR)])
                wU = max(1.0 - np.sum(np.clip(wR, 0, 1)), 0.0)
                ax.plot(x_plot, wU * _gauss(x_plot, muU, sigmaU),
                        color='blue', lw=1.0, ls='--',
                        label=f'unresolved (mu={muU:.4f})')
                for k in range(nR):
                    w = np.clip(p[2 + 4 * k], 0, 1)
                    mu = p[3 + 4 * k]
                    sig = p[4 + 4 * k]
                    alpha = p[5 + 4 * k]
                    ax.plot(x_plot, w * _skewnorm(x_plot, mu, sig, alpha),
                            color='orange', lw=1.0, ls=':',
                            label=f'resolved{k+1}' if i == 0 else None)
            elif mt == 'skewnorm':
                ax.plot(x_plot, _mixture_pdf_skewnorm(x_plot, p, nR),
                        color='red', lw=1.2, label='model')
                sigmaU = p[0]
                wR = np.array([p[1 + 4 * k] for k in range(nR)])
                wU = max(1.0 - np.sum(np.clip(wR, 0, 1)), 0.0)
                ax.plot(x_plot, wU * _gauss(x_plot, 0.0, sigmaU),
                        color='blue', lw=1.0, ls='--', label='unresolved')
                for k in range(nR):
                    w = np.clip(p[1 + 4 * k], 0, 1)
                    mu = p[2 + 4 * k]
                    sig = p[3 + 4 * k]
                    alpha = p[4 + 4 * k]
                    ax.plot(x_plot, w * _skewnorm(x_plot, mu, sig, alpha),
                            color='orange', lw=1.0, ls=':',
                            label=f'resolved{k+1}' if i == 0 else None)
            else:
                ax.plot(x_plot, _mixture_pdf(x_plot, p, nR),
                        color='red', lw=1.2, label='model')
                sigmaU = p[0]
                wR = np.array([p[1 + 3 * k] for k in range(nR)])
                wU = max(1.0 - np.sum(np.clip(wR, 0, 1)), 0.0)
                ax.plot(x_plot, wU * _gauss(x_plot, 0.0, sigmaU),
                        color='blue', lw=1.0, ls='--', label='unresolved')
                for k in range(nR):
                    w = np.clip(p[1 + 3 * k], 0, 1)
                    mu = p[2 + 3 * k]
                    sig = p[3 + 3 * k]
                    ax.plot(x_plot, w * _gauss(x_plot, mu, sig),
                            color='orange', lw=1.0, ls=':',
                            label=f'resolved{k+1}' if i == 0 else None)

        ax.set_xlim(dm_range)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if name is not None:
        plt.savefig(file_path + name, dpi=200)
    plt.show()
    return fig


# ===== Task 2: abstract CModel dependence =====

def _param_names(n_resolved):
    names = ['sigmaU']
    for k in range(n_resolved):
        names += [f'wR{k+1}', f'muR{k+1}', f'sigmaR{k+1}']
    return names


# Magnitude pivot used when fitting sigmaU(m). Fitting on x = m - MAG_PIVOT
# instead of raw m keeps the exponential term O(1) near survey depth and
# avoids numerical instabilities (Sesar+ 2007, eq. 5, p. 2238).
MAG_PIVOT = 25.0


def sesar_sigma(m, sigma0, a, b):
    """Sesar+ 2007 eq. 5 locus-width model, fit on x = m - MAG_PIVOT.

    sigma(m) = sqrt(sigma0^2 + a^2 * 10^(b * (m - MAG_PIVOT)))
    """
    x = np.asarray(m, dtype=float) - MAG_PIVOT
    return np.sqrt(sigma0 ** 2 + (a ** 2) * 10.0 ** (b * x))


def abstract_cmodel_dependence(fit_results, poly_deg=2, mode='fit'):
    """Abstract CModel dependence of the mixture parameters.

    Parameters
    ----------
    mode : {'fit', 'interp'}
        'fit'    — sigmaU via Sesar+ 2007 eq. 5, others via polynomial in CModel.
        'interp' — all parameters linearly interpolated between slice-fit values
                   (clamped at the endpoints). Useful when the smooth fit is
                   poor; falls back to per-slice Gaussian fits directly.
    """
    successful = [r for r in fit_results if r['success']]
    if not successful:
        raise RuntimeError('No successful slice fits to abstract.')

    n_resolved = successful[0]['n_resolved']
    names = _param_names(n_resolved)
    centers = np.array([r['cmodel_center'] for r in successful])
    values = {n: np.array([r['params'][i] for r in successful]) for i, n in enumerate(names)}

    sesar_params = None
    coeffs = {n: np.polyfit(centers, values[n], deg=poly_deg) for n in names}

    if mode == 'fit':
        # Fit sigmaU with Sesar eq. 5 using x = m - MAG_PIVOT.
        sU = values['sigmaU']
        p0 = (max(float(np.min(sU)), 1e-3), max(float(np.max(sU)), 1e-2), 0.4)
        try:
            sesar_params, _ = curve_fit(
                sesar_sigma, centers, sU, p0=p0,
                bounds=([1e-4, 1e-4, 0.0], [1.0, 5.0, 2.0]),
                maxfev=10000,
            )
        except Exception:
            sesar_params = None

    def func(cm):
        cm = np.atleast_1d(cm)
        out = np.zeros((len(names), len(cm)))
        if mode == 'interp':
            for i, n in enumerate(names):
                out[i] = np.interp(cm, centers, values[n])
        else:
            if sesar_params is not None:
                out[0] = sesar_sigma(cm, *sesar_params)
            else:
                out[0] = np.polyval(coeffs['sigmaU'], cm)
            for i, n in enumerate(names):
                if i == 0:
                    continue
                out[i] = np.polyval(coeffs[n], cm)
        out[0] = np.clip(out[0], 1e-3, 1.0)
        for k in range(n_resolved):
            out[1 + 3 * k] = np.clip(out[1 + 3 * k], 0.0, 1.0)
            out[3 + 3 * k] = np.clip(out[3 + 3 * k], 1e-3, 2.0)
        return out

    return {
        'n_resolved': n_resolved,
        'poly_deg': poly_deg,
        'mag_pivot': MAG_PIVOT,
        'mode': mode,
        'coeffs': coeffs,
        'sesar_params': sesar_params,
        'centers': centers,
        'values': values,
        'func': func,
        'names': names,
    }


def build_binned_model(fit_results):
    """Build an abs_model-compatible dict using raw per-bin parameters.

    Instead of fitting smooth functions through the slice-fit values,
    each source gets the parameters of whichever CModel bin it falls
    into (flat lookup, no interpolation or smoothing).
    """
    successful = [r for r in fit_results if r['success']]
    if not successful:
        raise RuntimeError('No successful slice fits to build binned model.')

    n_resolved = successful[0]['n_resolved']
    model_type = successful[0].get('model_type', 'gaussian')
    free_star_mean = successful[0].get('free_star_mean', False)

    if model_type == 'skewnorm' and free_star_mean:
        names = _param_names_skewnorm_freemu(n_resolved)
        stride = 4
        star_offset = 2
    elif model_type == 'skewnorm':
        names = _param_names_skewnorm(n_resolved)
        stride = 4
        star_offset = 1
    else:
        names = _param_names(n_resolved)
        stride = 3
        star_offset = 1

    bin_edges = np.array([(r['cmodel_lo'], r['cmodel_hi']) for r in successful])
    bin_params = np.array([r['params'] for r in successful])  # (n_bins, P)
    centers = np.array([r['cmodel_center'] for r in successful])
    values = {n: bin_params[:, i] for i, n in enumerate(names)}

    def func(cm):
        cm = np.atleast_1d(cm)
        out = np.zeros((len(names), len(cm)))
        for j, c in enumerate(cm):
            idx = np.searchsorted(bin_edges[:, 0], c, side='right') - 1
            idx = np.clip(idx, 0, len(bin_edges) - 1)
            out[:, j] = bin_params[idx]
        for k in range(n_resolved):
            out[star_offset + stride * k] = np.clip(out[star_offset + stride * k], 0.0, 1.0)
            out[star_offset + 2 + stride * k] = np.clip(out[star_offset + 2 + stride * k], 1e-3, 2.0)
        sigma_idx = 1 if free_star_mean else 0
        out[sigma_idx] = np.clip(out[sigma_idx], 1e-3, 1.0)
        return out

    return {
        'n_resolved': n_resolved,
        'model_type': model_type,
        'free_star_mean': free_star_mean,
        'centers': centers,
        'values': values,
        'func': func,
        'names': names,
        'mode': 'binned',
    }


def plot_param_curves(abs_model, name=None, file_path='../plots/'):
    names = abs_model['names']
    centers = abs_model['centers']
    values = abs_model['values']
    func = abs_model['func']

    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    cm_smooth = np.linspace(centers.min(), centers.max(), 200)
    smooth_vals = func(cm_smooth)

    sesar_params = abs_model.get('sesar_params')
    for i, n in enumerate(names):
        ax = axes[i]
        ax.plot(centers, values[n], 'ko', label='slice fits')
        fit_label = 'Sesar eq. 5 fit' if (n == 'sigmaU' and sesar_params is not None) else 'poly fit'
        ax.plot(cm_smooth, smooth_vals[i], 'r-', lw=1.2, label=fit_label)
        ax.set_xlabel('CModel mag')
        ax.set_ylabel(n)
        ax.grid(True, alpha=0.3)
        if n == 'sigmaU' and sesar_params is not None:
            s0, a, b = sesar_params
            ax.set_title(
                rf'$\sigma_0$={s0:.3f}, a={a:.3f}, b={b:.2f}  (x = m$-${abs_model["mag_pivot"]:.0f})',
                fontsize=9,
            )
        ax.legend(fontsize=8)

    for j in range(len(names), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if name is not None:
        plt.savefig(file_path + name, dpi=200)
    plt.show()
    return fig


# ===== Task 3: 2D model & residuals =====

def evaluate_2d_model(abs_model, dm_grid, cmodel_grid):
    """Return 2D model density evaluated on the grid.

    Shape: (len(cmodel_grid), len(dm_grid)) — rows are CModel, cols are PSF-CModel.
    Each row is a PDF over PSF-CModel at that CModel value.
    """
    n_resolved = abs_model['n_resolved']
    model_type = abs_model.get('model_type', 'gaussian')
    free_star_mean = abs_model.get('free_star_mean', False)
    if model_type == 'skewnorm' and free_star_mean:
        pdf_func = _mixture_pdf_skewnorm_freemu
    elif model_type == 'skewnorm':
        pdf_func = _mixture_pdf_skewnorm
    else:
        pdf_func = _mixture_pdf
    params_cm = abs_model['func'](cmodel_grid)  # shape (P, len(cmodel_grid))
    out = np.zeros((len(cmodel_grid), len(dm_grid)))
    for j in range(len(cmodel_grid)):
        p = params_cm[:, j]
        out[j] = pdf_func(dm_grid, p, n_resolved)
    return out


def _two_d_histogram(df, filter_band, dm_range, cm_range, nbins):
    dm_col = f'{filter_band}{model_flux_diff}'
    cm_col = f'{filter_band}{model_flux_mag}'
    mask = (
        np.isfinite(df[dm_col])
        & np.isfinite(df[cm_col])
        & (df[dm_col] > dm_range[0])
        & (df[dm_col] < dm_range[1])
        & (df[cm_col] > cm_range[0])
        & (df[cm_col] < cm_range[1])
    )
    H, xedges, yedges = np.histogram2d(
        df.loc[mask, dm_col].to_numpy(),
        df.loc[mask, cm_col].to_numpy(),
        bins=nbins,
        range=[dm_range, cm_range],
    )
    return H, xedges, yedges, mask.sum()


def plot_2d_model_residuals(
    df,
    filter_band,
    abs_model,
    dm_range=DEFAULT_DM_RANGE,
    cm_range=DEFAULT_CMODEL_RANGE,
    nbins=(120, 80),
    name=None,
    file_path='../plots/',
    cmap='cividis',
):
    H, xedges, yedges, n_used = _two_d_histogram(df, filter_band, dm_range, cm_range, nbins)
    # H is (ndm, ncm); transpose so rows are CModel
    H = H.T

    dm_centers = 0.5 * (xedges[:-1] + xedges[1:])
    cm_centers = 0.5 * (yedges[:-1] + yedges[1:])

    model_pdf = evaluate_2d_model(abs_model, dm_centers, cm_centers)

    # normalize model per CModel row to match data per-row counts
    dm_width = xedges[1] - xedges[0]
    row_counts = H.sum(axis=1, keepdims=True)
    row_norm = model_pdf * dm_width
    row_norm_sum = row_norm.sum(axis=1, keepdims=True)
    row_norm_sum = np.where(row_norm_sum > 0, row_norm_sum, 1.0)
    model_counts = row_counts * row_norm / row_norm_sum

    resid = H - model_counts

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    extent = [dm_range[0], dm_range[1], cm_range[0], cm_range[1]]

    vmax = max(H.max(), model_counts.max())
    im0 = axes[0].imshow(H, origin='lower', extent=extent, aspect='auto',
                         cmap=cmap, vmin=0, vmax=vmax)
    axes[0].set_title(f'{filter_band}: data')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('PSF - CModel')
    axes[0].set_ylabel('CModel mag')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(model_counts, origin='lower', extent=extent, aspect='auto',
                         cmap=cmap, vmin=0, vmax=vmax)
    axes[1].set_title(f'{filter_band}: model')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('PSF - CModel')
    axes[1].set_ylabel('CModel mag')
    fig.colorbar(im1, ax=axes[1])

    rlim = np.nanpercentile(np.abs(resid), 98) if resid.size else 1.0
    im2 = axes[2].imshow(resid, origin='lower', extent=extent, aspect='auto',
                         cmap='RdBu_r', vmin=-rlim, vmax=rlim)
    axes[2].set_title(f'{filter_band}: data - model')
    axes[2].invert_yaxis()
    axes[2].set_xlabel('PSF - CModel')
    axes[2].set_ylabel('CModel mag')
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if name is not None:
        plt.savefig(file_path + name, dpi=200)
    plt.show()
    return {'H': H, 'model': model_counts, 'resid': resid,
            'dm_centers': dm_centers, 'cm_centers': cm_centers}


# ===== Task 4: pS (unresolved probability) =====

def compute_pS(dm_array, cmodel_array, abs_model):
    """Unresolved-component fraction at each (PSF-CModel, CModel) point."""
    dm_array = np.asarray(dm_array, dtype=float)
    cmodel_array = np.asarray(cmodel_array, dtype=float)
    n_resolved = abs_model['n_resolved']
    model_type = abs_model.get('model_type', 'gaussian')
    free_star_mean = abs_model.get('free_star_mean', False)

    params_pt = abs_model['func'](cmodel_array)  # shape (P, N)

    if free_star_mean:
        muU = params_pt[0]
        sigmaU = params_pt[1]
        star_offset = 2
    else:
        muU = 0.0
        sigmaU = params_pt[0]
        star_offset = 1

    if model_type == 'skewnorm':
        stride = 4
    else:
        stride = 3

    wR = np.array([params_pt[star_offset + stride * k] for k in range(n_resolved)])
    wR = np.clip(wR, 0.0, 1.0)
    wU = np.clip(1.0 - wR.sum(axis=0), 0.0, 1.0)

    p_unres = wU * _gauss(dm_array, muU, sigmaU)

    p_total = p_unres.copy()
    for k in range(n_resolved):
        mu = params_pt[star_offset + 1 + stride * k]
        sig = params_pt[star_offset + 2 + stride * k]
        if model_type == 'skewnorm':
            alpha = params_pt[star_offset + 3 + stride * k]
            p_total = p_total + wR[k] * _skewnorm(dm_array, mu, sig, alpha)
        else:
            p_total = p_total + wR[k] * _gauss(dm_array, mu, sig)

    with np.errstate(invalid='ignore', divide='ignore'):
        pS = np.where(p_total > 0, p_unres / p_total, np.nan)
    return pS


def plot_pS_map(
    df,
    filter_band,
    abs_model,
    dm_range=DEFAULT_DM_RANGE,
    cm_range=DEFAULT_CMODEL_RANGE,
    nbins=(120, 80),
    name=None,
    file_path='../plots/',
    cmap='coolwarm_r',
):
    """Median pS per pixel in the CModel vs PSF-CModel diagram."""
    dm_col = f'{filter_band}{model_flux_diff}'
    cm_col = f'{filter_band}{model_flux_mag}'
    mask = (
        np.isfinite(df[dm_col])
        & np.isfinite(df[cm_col])
        & (df[dm_col] > dm_range[0])
        & (df[dm_col] < dm_range[1])
        & (df[cm_col] > cm_range[0])
        & (df[cm_col] < cm_range[1])
    )
    dm_vals = df.loc[mask, dm_col].to_numpy()
    cm_vals = df.loc[mask, cm_col].to_numpy()

    pS = compute_pS(dm_vals, cm_vals, abs_model)

    xedges = np.linspace(dm_range[0], dm_range[1], nbins[0] + 1)
    yedges = np.linspace(cm_range[0], cm_range[1], nbins[1] + 1)
    ix = np.clip(np.digitize(dm_vals, xedges) - 1, 0, nbins[0] - 1)
    iy = np.clip(np.digitize(cm_vals, yedges) - 1, 0, nbins[1] - 1)

    img = np.full((nbins[1], nbins[0]), np.nan)
    df_pix = pd.DataFrame({'ix': ix, 'iy': iy, 'pS': pS}).dropna()
    grouped = df_pix.groupby(['iy', 'ix'])['pS'].median()
    for (iyv, ixv), val in grouped.items():
        img[iyv, ixv] = val

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    extent = [dm_range[0], dm_range[1], cm_range[0], cm_range[1]]
    im = ax.imshow(img, origin='lower', extent=extent, aspect='auto',
                   cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f'{filter_band}: median pS (unresolved fraction) per pixel')
    ax.invert_yaxis()
    ax.set_xlabel('PSF - CModel')
    ax.set_ylabel('CModel mag')
    fig.colorbar(im, ax=ax, label='pS')

    plt.tight_layout()
    if name is not None:
        plt.savefig(file_path + name, dpi=200)
    plt.show()
    return img

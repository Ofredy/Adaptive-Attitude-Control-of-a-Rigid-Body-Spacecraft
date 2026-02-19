import os

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_inertia_estimate_history(
    mrp_sum: np.ndarray,
    dt: float,
    J_actual: np.ndarray,
    estimated_inertia_t0: np.ndarray, 
    out_dir: str,
    theta_start_idx: int = 12,
    prefix: str = "att_track",
    units: str = "kg*m^2",
    y_window_frac: float = None,   # +/- 2% window around truth (set None to disable)
    skip_seconds: float = 0.0,     # skip initial transient time for nicer plots
    save_error_plots: bool = True,
):
    """
    Plot estimated diagonal inertia history vs actual inertia (and optionally error).

    Fixes Matplotlib "weird scaling" by disabling y-axis offset notation.

    Assumes mrp_sum is the state history with shape (N, nx) (preferred).
    State layout (learn_inertia=True):
        [0:3]   mrp_b_n
        [3:6]   w_b_n
        [6:9]   state_sum
        [9:12]  (whatever constant slot you keep, e.g. w_b_r_0)
        [12:15] theta_hat = [Jx, Jy, Jz]

    Args:
        mrp_sum: (N, nx) or (nx, N) numpy array of state history
        dt: timestep [s]
        J_actual: (3,3) actual inertia tensor
        out_dir: base directory to save plots
        theta_start_idx: start index for theta_hat in state vector (default 12)
        prefix: filename prefix for saved plots
        units: label for y-axis
        y_window_frac: if not None, sets y-lims to (1±y_window_frac)*J_true per axis
        skip_seconds: skip first skip_seconds of data in plots
        save_error_plots: also save J_hat - J_true plots

    Saves:
        out_dir/plots/inertia/<prefix>_Jx.png, Jy.png, Jz.png, combined.png
        out_dir/plots/inertia_error/<prefix>_Jx_err.png, ...
    """
    X = np.asarray(mrp_sum)

    if X.ndim != 2:
        raise ValueError("mrp_sum must be a 2D numpy array.")

    # Ensure shape is (N, nx); if (nx, N), transpose
    if X.shape[0] in (12, 15, 18) and X.shape[1] > X.shape[0]:
        X = X.T

    N, nx = X.shape
    theta_slice = slice(theta_start_idx, theta_start_idx + 3)

    if nx < theta_start_idx + 3:
        raise ValueError(
            f"State dimension nx={nx} too small for theta indices "
            f"[{theta_start_idx}:{theta_start_idx+3}]."
        )

    t = np.arange(N) * dt
    theta_hist = X[:, theta_slice]  # (N,3)
    theta_hist = np.diag(estimated_inertia_t0) + theta_hist

    J_actual_diag = np.diag(np.asarray(J_actual)).reshape(3,)

    inertia_dir = os.path.join(out_dir, "plots", "inertia")
    _ensure_dir(inertia_dir)

    err_dir = os.path.join(out_dir, "plots", "inertia_error")
    if save_error_plots:
        _ensure_dir(err_dir)

    names = ["Jx", "Jy", "Jz"]

    # skip initial transient
    skip = int(skip_seconds / dt) if skip_seconds and skip_seconds > 0 else 0
    t_plot = t[skip:]

    # Individual plots
    for k, name in enumerate(names):
        J_true = J_actual_diag[k]
        J_hat = theta_hist[:, k]
        J_hat_plot = J_hat[skip:]
        J_true_plot = np.full_like(t_plot, J_true)

        fig, ax = plt.subplots()
        ax.plot(t_plot, J_hat_plot, label="Estimated", linewidth=2)
        ax.plot(t_plot, J_true_plot, "--", label="Actual", linewidth=2)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{name} [{units}]" if units else name)
        ax.set_title(f"{name}: estimated vs actual")
        ax.grid(True)
        ax.legend()

        # IMPORTANT: disable offset notation that causes "1e-10 + 1e2"
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

        # Optional: lock y-limits around truth so it looks clean
        if y_window_frac is not None:
            # handle J_true near zero robustly
            base = abs(J_true) if abs(J_true) > 1e-12 else 1.0
            margin = y_window_frac * base
            ax.set_ylim(J_true - margin, J_true + margin)

        fig.savefig(os.path.join(inertia_dir, f"{prefix}_{name}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Error plots
        if save_error_plots:
            err = (J_hat - J_true)[skip:]

            fig, ax = plt.subplots()
            ax.plot(t_plot, err, linewidth=2)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{name} error [{units}]" if units else f"{name} error")
            ax.set_title(f"{name} estimation error (Estimated - Actual)")
            ax.grid(True)

            ax.ticklabel_format(style="plain", axis="y", useOffset=False)

            fig.savefig(os.path.join(err_dir, f"{prefix}_{name}_err.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    # Combined plot
    fig, ax = plt.subplots()
    for k, name in enumerate(names):
        ax.plot(t_plot, theta_hist[skip:, k], label=f"{name} est", linewidth=2)
        ax.plot(t_plot, np.full_like(t_plot, J_actual_diag[k]), "--", label=f"{name} actual", linewidth=2)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Inertia [{units}]" if units else "Inertia")
    ax.set_title("Diagonal inertia: estimated vs actual")
    ax.grid(True)
    ax.legend(ncols=2)

    ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    fig.savefig(os.path.join(inertia_dir, f"{prefix}_combined.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return inertia_dir

def plot_mrp_error_history(
    mrp_err_hist: np.ndarray,
    dt: float,
    out_dir: str,
    prefix: str = "att_track",
    skip_seconds: float = 0.0,
):
    """
    Plot MRP tracking error sigma_{B/R} over time.

    Accepts:
      - mrp_err_hist shape (N,3) or (3,N)
    Saves into:
      out_dir/plots/mrp_error/
    """
    S = np.asarray(mrp_err_hist)

    if S.ndim != 2:
        raise ValueError("mrp_err_hist must be a 2D array with shape (N,3) or (3,N).")

    # allow (3,N)
    if S.shape[0] == 3 and S.shape[1] != 3:
        S = S.T  # -> (N,3)

    if S.shape[1] != 3:
        raise ValueError(f"mrp_err_hist must have 3 columns (got shape {S.shape}).")

    N = S.shape[0]
    t = np.arange(N) * dt

    err_dir = os.path.join(out_dir, "plots", "mrp_error")
    _ensure_dir(err_dir)

    skip = int(skip_seconds / dt) if skip_seconds and skip_seconds > 0 else 0
    t_plot = t[skip:]
    S_plot = S[skip:, :]
    sig_norm = np.linalg.norm(S, axis=1)[skip:]

    # sigma1/2/3
    for k, name in enumerate(["sigma1", "sigma2", "sigma3"]):
        fig, ax = plt.subplots()
        ax.plot(t_plot, S_plot[:, k], linewidth=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(name)
        ax.set_title(f"{name} tracking error (MRP)")
        ax.grid(True)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        fig.savefig(os.path.join(err_dir, f"{prefix}_{name}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # norm
    fig, ax = plt.subplots()
    ax.plot(t_plot, sig_norm, linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("||sigma||")
    ax.set_title("Attitude tracking error norm (MRP)")
    ax.grid(True)
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    fig.savefig(os.path.join(err_dir, f"{prefix}_sigma_norm.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # components overlay
    fig, ax = plt.subplots()
    ax.plot(t_plot, S_plot[:, 0], label="sigma1", linewidth=2)
    ax.plot(t_plot, S_plot[:, 1], label="sigma2", linewidth=2)
    ax.plot(t_plot, S_plot[:, 2], label="sigma3", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("sigma components")
    ax.set_title("Attitude tracking error components (MRP)")
    ax.grid(True)
    ax.legend()
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    fig.savefig(os.path.join(err_dir, f"{prefix}_sigma_components.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return err_dir

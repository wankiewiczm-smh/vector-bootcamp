import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_nbm_shape_functions_with_feature_density(
    model,
    X,
    feature_names=None,
    n_points=50,
    bins=50,
    plot_cols=4,
    figsize=None,
    device="cpu",
    red_alpha=0.6,
):
    """
    Plot partial dependence curves (blue) for each feature in a Neural Basis Model,
    with vertical red shading for the feature distribution.

    This refactored version ensures each subplot cell has the same size
    using constrained_layout=True.

    Args:
        model (nn.Module):
            A trained NBM model in eval mode that outputs a single value
            per sample (e.g., for regression).
        X (np.ndarray or torch.Tensor):
            Feature matrix of shape (N, D). The model should accept X as input.
        feature_names (list[str]):
            Optional names for the D features.
        n_points (int):
            Number of points in the grid for partial dependence.
        bins (int):
            Number of bins for the 1D histogram used in the red shading.
        plot_cols (int):
            Number of columns in the subplot grid.
        figsize (tuple):
            Overall figure size (width, height) in inches. If None, a default is used.
        device (str):
            "cpu" or "cuda" for the model and data.
        red_alpha (float):
            Transparency for the red shading (0.0=transparent, 1.0=opaque).
    """
    model.eval()

    # Convert input X to a torch tensor if needed, and move to device
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    # Determine the number of rows/columns for subplots
    ncols = min(plot_cols, n_features)
    nrows = math.ceil(n_features / ncols)

    # If user didn't provide a figure size, pick a default
    if figsize is None:
        # Increase these values if you want larger subplots
        figsize = (2 * ncols, 2 * nrows)

    # Use constrained_layout so each subplot cell has the same size
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=True
    )

    # Flatten axes in case there's only one row or one column
    axes = np.array(axes).ravel()

    for j in range(n_features):
        ax = axes[j]

        # 1) Create a grid of feature values in [min, max]
        xj_min = X[:, j].min().item()
        xj_max = X[:, j].max().item()
        xj_vals = torch.linspace(xj_min, xj_max, steps=n_points, device=device)

        # 2) Compute ICE (Individual Conditional Expectation) for each sample
        ice_curves = []
        for i in range(n_samples):
            row = X[i].clone()
            row_expanded = row.unsqueeze(0).repeat(n_points, 1)
            row_expanded[:, j] = xj_vals  # vary only feature j

            with torch.no_grad():
                preds = model(row_expanded)
            ice_curves.append(preds.cpu().numpy().flatten())

        ice_curves = np.array(ice_curves)  # shape (n_samples, n_points)
        pdp = ice_curves.mean(axis=0)      # partial dependence is mean over samples

        # 3) Build the vertical red shading from a 1D histogram of X[:, j]
        feature_values = X[:, j].cpu().numpy()
        hist, bin_edges = np.histogram(feature_values, bins=bins, range=(xj_min, xj_max))
        hist = hist.astype(float)
        if hist.max() > 0:
            hist /= hist.max()  # normalize to [0, 1]

        # Compute y-limits, with a bit of padding
        y_min, y_max = pdp.min(), pdp.max()
        padding = 0.1 * (y_max - y_min) if (y_max - y_min) > 0 else 0.1
        y_min -= padding
        y_max += padding

        # Repeat the histogram vertically to form a 2D shading
        height = 100
        hist_2d = np.tile(hist, (height, 1))

        ax.imshow(
            hist_2d,
            extent=(bin_edges[0], bin_edges[-1], y_min, y_max),
            cmap="Reds",
            alpha=red_alpha,
            aspect="auto",
            origin="lower",
        )

        # 4) Plot the partial dependence (blue line)
        ax.plot(xj_vals.cpu().numpy(), pdp, color="blue", linewidth=2)

        # 5) Labeling
        ax.set_title(feature_names[j])
        ax.set_xlabel(feature_names[j])
        ax.set_ylabel("Model Output")
        ax.set_xlim([xj_min, xj_max])
        ax.set_ylim([y_min, y_max])

    # Hide any extra subplot cells if we have fewer features than nrows*ncols
    for k in range(n_features, len(axes)):
        axes[k].axis("off")

    plt.show()
    
    
    
# use permutation_importance and plot_feature_importance functions to plot feature importance

def permutation_importance(
    model,
    X,
    y,
    metric=mean_squared_error,
    n_repeats=5,
    random_state=42
):
    """
    Compute permutation feature importance for a regression model using MSE drop.

    Args:
        model (nn.Module): Trained PyTorch model (in eval mode).
        X (torch.Tensor or np.ndarray): Input features of shape (N, D).
        y (torch.Tensor or np.ndarray): Target values of shape (N,) or (N, 1).
        metric (callable): A function f(preds, y) -> float to measure performance.
                           Defaults to mean_squared_error.
        n_repeats (int): How many times to shuffle each feature for stability.
        random_state (int): Random seed for reproducibility.

    Returns:
        importances (np.ndarray): Mean drop in performance for each feature (D,).
        importances_std (np.ndarray): Standard deviation of drop in performance (D,).
    """
    # Convert X, y to torch.Tensor if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if y.dim() == 2 and y.shape[1] == 1:
        y = y.squeeze(-1)  # make y shape (N,)

    # Move to same device as model if necessary
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)

    # Baseline performance
    model.eval()
    with torch.no_grad():
        baseline_preds = model(X).squeeze(-1)  # shape (N,)
    baseline_score = metric(baseline_preds.cpu().numpy(), y.cpu().numpy())

    # For each feature, shuffle values and measure performance drop
    rng = np.random.default_rng(seed=random_state)
    n_features = X.shape[1]
    importances = np.zeros(n_features)
    importances_sq = np.zeros(n_features)  # for variance calculation

    # We do multiple repeats per feature
    for rep in range(n_repeats):
        for j in range(n_features):
            # Save a copy of the original column
            original_col = X[:, j].clone()

            # Shuffle this feature column in-place
            perm_indices = rng.permutation(X.shape[0])
            X[:, j] = X[perm_indices, j]

            # Evaluate performance
            with torch.no_grad():
                perm_preds = model(X).squeeze(-1)
            perm_score = metric(perm_preds.cpu().numpy(), y.cpu().numpy())

            # Restore the original column
            X[:, j] = original_col

            # Compute the drop in performance (for MSE, bigger = more important)
            score_drop = perm_score - baseline_score
            importances[j] += score_drop
            importances_sq[j] += score_drop ** 2

    # Average over repeats
    importances /= float(n_repeats)
    importances_sq /= float(n_repeats)
    importances_std = np.sqrt(importances_sq - importances**2)

    return importances, importances_std


def plot_feature_importance(importances, importances_std, feature_names=None):
    """
    Plot permutation feature importances as a bar chart.

    Args:
        importances (np.ndarray): Mean drop in performance for each feature (D,).
        importances_std (np.ndarray): Standard deviation of drop for each feature (D,).
        feature_names (list[str]): Names for each feature.
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort by importance
    idx_sorted = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx_sorted]
    sorted_importances = importances[idx_sorted]
    sorted_std = importances_std[idx_sorted]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), sorted_importances, yerr=sorted_std, align='center')
    plt.xticks(range(len(importances)), sorted_names, rotation=45, ha='right')
    plt.ylabel("Importance (Performance Drop)")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.show()
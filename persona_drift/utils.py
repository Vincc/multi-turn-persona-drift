# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


def _to_numpy(x):
    """Convert tensor or array to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)}")


class MeanScaler:
    """Scaler that centers data by subtracting the mean."""

    def __init__(self, mean=None):
        self.mean = mean

    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)

    def fit(self, X):
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))
            self.mean = X_np.mean(axis=axes, keepdims=False)
        else:
            self._ensure_mean_numpy()
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("MeanScaler not fitted")
        self._ensure_mean_numpy()
        return _to_numpy(X) - self.mean

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def compute_pca(activations_2d, scaler):
    """
    Fit PCA on 2D activations (n_roles, d_model), after centering with scaler.

    Returns (pca, fitted_scaler, variance_explained).
    """
    scaled = scaler.fit_transform(activations_2d)
    X_np = _to_numpy(scaled)

    pca = PCA()
    pca.fit(X_np)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    print(f"PCA fitted with {len(variance_explained)} components; "
          f"cumulative variance for first 5: {cumulative_variance[:5]}")

    return pca, scaler, variance_explained


def project_onto_pcs(vector, pca, scaler, component=None):
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().float().numpy()
    vector = np.asarray(vector, dtype=np.float64)[None, :]
    scaled = scaler.transform(vector)
    projected = pca.transform(scaled)
    if component is not None:
        projected = projected[:, component]
    return projected[0]


def project_topk(vector, pca, scaler, k):
    """Project vector onto the first k principal components. Returns list[float]."""
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().float().numpy()
    vector = np.asarray(vector, dtype=np.float64)[None, :]
    scaled = scaler.transform(vector)
    projected = pca.transform(scaled)[0, :k]
    return projected.tolist()


def load_persona_space(vector_model, target_layer, repo_id):
    """
    Download Lu et al.'s role vectors and fit a persona PC space. Code from assistant-axis repo
    """

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{vector_model}/role_vectors/*.pt",
                         f"{vector_model}/default_vector.pt"],
    )
    role_vectors = {p.stem: torch.load(p, map_location="cpu", weights_only=False)
                     for p in Path(local_dir, vector_model, "role_vectors").glob("*.pt")}
    role_vectors_at_layer = torch.stack(
        [v[target_layer] for v in role_vectors.values()]).float()

    pca, scaler, variance_explained = compute_pca(role_vectors_at_layer, MeanScaler())

    return {
        "pca": pca,
        "scaler": scaler,
        "variance_explained": variance_explained,
        "role_labels": list(role_vectors.keys()),
    }


class ActivationCache:
    def __init__(self):
        self.acts = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.acts = hs.detach().cpu()


def load_model(model_name):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("tokenizer loaded")
    kwargs = {"device_map": "auto", "torch_dtype": "auto"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print("model loaded")
    return model, tokenizer

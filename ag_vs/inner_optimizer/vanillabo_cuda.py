"""CUDA posterior adapter for the fitted sklearn AG-VS-BO Gaussian process."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product, Sum, WhiteKernel


def _fitted_kernel_parameters(kernel) -> Tuple[float, np.ndarray, float]:
    """Extract Constant x Matern-5/2 and optional WhiteKernel parameters."""
    white_noise = 0.0
    signal = kernel
    if isinstance(kernel, Sum):
        if isinstance(kernel.k1, WhiteKernel):
            white_noise = float(kernel.k1.noise_level)
            signal = kernel.k2
        elif isinstance(kernel.k2, WhiteKernel):
            white_noise = float(kernel.k2.noise_level)
            signal = kernel.k1
    if not isinstance(signal, Product):
        raise TypeError("AG-VS-BO CUDA requires a fitted ConstantKernel x Matern kernel")
    if isinstance(signal.k1, ConstantKernel) and isinstance(signal.k2, Matern):
        constant, matern = signal.k1, signal.k2
    elif isinstance(signal.k2, ConstantKernel) and isinstance(signal.k1, Matern):
        constant, matern = signal.k2, signal.k1
    else:
        raise TypeError("AG-VS-BO CUDA requires a fitted ConstantKernel x Matern kernel")
    if not np.isclose(float(matern.nu), 2.5):
        raise ValueError("AG-VS-BO CUDA supports only the configured Matern-5/2 kernel")
    return (
        float(constant.constant_value),
        np.asarray(matern.length_scale, dtype=float).reshape(-1),
        float(white_noise),
    )


class CudaSklearnGPPredictor:
    """Sklearn-compatible GP predictor backed by torch CUDA linear algebra.

    Hyperparameters and the Cholesky factor are still produced by the canonical
    sklearn fit. Only posterior prediction is moved to CUDA, preserving the
    fitted model and analytic-EI semantics used by AG-VS-BO.
    """

    def __init__(self, fitted_model, *, device: str = "cuda") -> None:
        import torch

        device_name = str(device).lower()
        if device_name not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("AG-VS-BO CUDA prediction requires an available CUDA device")
        self._sklearn_model = fitted_model
        self.device = torch.device(device_name)
        self.dtype = torch.float64
        self.X_train_ = np.asarray(fitted_model.X_train_, dtype=float)
        self.y_train_ = np.asarray(fitted_model.y_train_, dtype=float)
        self.kernel_ = fitted_model.kernel_
        self.L_ = np.asarray(fitted_model.L_, dtype=float)
        self.alpha_ = np.asarray(fitted_model.alpha_, dtype=float)
        self.log_marginal_likelihood_value_ = getattr(
            fitted_model, "log_marginal_likelihood_value_", float("nan")
        )
        self._y_train_mean = np.asarray(
            getattr(fitted_model, "_y_train_mean", 0.0), dtype=float
        )
        self._y_train_std = np.asarray(
            getattr(fitted_model, "_y_train_std", 1.0), dtype=float
        )
        constant, lengthscales, white_noise = _fitted_kernel_parameters(self.kernel_)
        if lengthscales.size == 1 and self.X_train_.shape[1] > 1:
            lengthscales = np.full(self.X_train_.shape[1], float(lengthscales[0]))
        if lengthscales.size != self.X_train_.shape[1]:
            raise ValueError("Fitted kernel length scales do not match the GP input dimension")
        self._constant = float(constant)
        self._white_noise = float(white_noise)
        self._train_t = torch.as_tensor(
            self.X_train_, device=self.device, dtype=self.dtype
        )
        self._lengthscales_t = torch.as_tensor(
            lengthscales, device=self.device, dtype=self.dtype
        )
        self._alpha_t = torch.as_tensor(
            self.alpha_, device=self.device, dtype=self.dtype
        )
        self._L_t = torch.as_tensor(self.L_, device=self.device, dtype=self.dtype)

    def __getattr__(self, name):
        return getattr(self._sklearn_model, name)

    def _signal_kernel(self, left, right):
        import torch

        scaled = (left[:, None, :] - right[None, :, :]) / self._lengthscales_t
        distance = torch.sqrt(torch.clamp(torch.sum(scaled * scaled, dim=-1), min=0.0))
        sqrt_five_distance = np.sqrt(5.0) * distance
        return self._constant * (
            1.0 + sqrt_five_distance + (5.0 / 3.0) * distance * distance
        ) * torch.exp(-sqrt_five_distance)

    def predict(self, X, return_std: bool = False, return_cov: bool = False):
        import torch

        if return_std and return_cov:
            raise RuntimeError("At most one of return_std or return_cov can be requested")
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"Expected X with shape (n, {self.X_train_.shape[1]}), got {X_array.shape}"
            )
        query_t = torch.as_tensor(X_array, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            cross = self._signal_kernel(query_t, self._train_t)
            mean_t = cross @ self._alpha_t
            mean = mean_t.detach().cpu().numpy()
            mean = mean * self._y_train_std + self._y_train_mean
            if not return_std and not return_cov:
                return np.asarray(mean, dtype=float)

            right_hand_side = cross.transpose(0, 1)
            if hasattr(torch.linalg, "solve_triangular"):
                solved = torch.linalg.solve_triangular(
                    self._L_t,
                    right_hand_side,
                    upper=False,
                )
            else:
                solved = torch.triangular_solve(
                    right_hand_side,
                    self._L_t,
                    upper=False,
                )[0]
            if return_cov:
                covariance_t = self._signal_kernel(query_t, query_t)
                covariance_t = covariance_t - solved.transpose(0, 1) @ solved
                if self._white_noise > 0.0:
                    covariance_t.diagonal().add_(self._white_noise)
                covariance_t = 0.5 * (
                    covariance_t + covariance_t.transpose(0, 1)
                )
                covariance_t.diagonal().clamp_(min=0.0)
                covariance = covariance_t.detach().cpu().numpy()
                covariance = covariance * np.square(self._y_train_std)
                return np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)

            variance_t = self._constant + self._white_noise - torch.sum(
                solved * solved, dim=0
            )
            variance_t = torch.clamp(variance_t, min=0.0)
            std = torch.sqrt(variance_t).detach().cpu().numpy()
            std = std * self._y_train_std
            return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def cuda_wrap_sklearn_gp(fitted_model):
    return CudaSklearnGPPredictor(fitted_model, device="cuda")


__all__ = ["CudaSklearnGPPredictor", "cuda_wrap_sklearn_gp"]

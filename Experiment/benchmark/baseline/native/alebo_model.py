# Copyright (c) Facebook, Inc. and its affiliates. MIT license.
# Ported from Ax 0.2.2: ALEBO Mahalanobis kernel and moment-matched GP.
from typing import Any, Optional, List, Union
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

class ALEBOKernel(Kernel):
    """The kernel for ALEBO.

    Suppose there exists an ARD RBF GP on an (unknown) linear embedding with
    projection matrix A. We make function evaluations in a different linear
    embedding with projection matrix B (known). This is the appropriate kernel
    for fitting those data.

    This kernel computes a Mahalanobis distance, and the (d x d) PD distance
    matrix Gamma is a parameter that must be fit. This is done by fitting its
    upper Cholesky decomposition, U.

    Args:
        B: (d x D) Projection matrix.
        batch_shape: Batch shape as usual for gpytorch kernels.
    """

    def __init__(self, B: Tensor, batch_shape: torch.Size) -> None:
        super().__init__(
            has_lengthscale=False, ard_num_dims=None, eps=0.0, batch_shape=batch_shape
        )
        self.d, D = B.shape
        assert self.d <= D
        self.B = B
        # Initialize U
        Arnd = torch.randn(D, D, dtype=B.dtype, device=B.device)
        Arnd = torch.linalg.qr(Arnd)[0]
        ABinv = Arnd[: self.d, :] @ torch.pinverse(B)
        # U is the upper Cholesky decomposition of Gamma, the Mahalanobis
        # matrix. Uvec is the upper triangular portion of U squeezed out into
        # a vector.
        U = torch.linalg.cholesky(torch.mm(ABinv.t(), ABinv)).mT
        self.triu_indx = torch.triu_indices(self.d, self.d, device=B.device)
        Uvec = U[self.triu_indx[0], self.triu_indx[1]].repeat(*batch_shape, 1)
        self.register_parameter(name="Uvec", parameter=torch.nn.Parameter(Uvec))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        """Compute kernel distance."""
        # Unpack Uvec into an upper triangular matrix U
        shapeU = self.Uvec.shape[:-1] + torch.Size([self.d, self.d])
        U_t = torch.zeros(shapeU, dtype=self.B.dtype, device=self.B.device)
        U_t[..., self.triu_indx[1], self.triu_indx[0]] = self.Uvec
        # Compute kernel distance
        z1 = torch.matmul(x1, U_t)
        z2 = torch.matmul(x2, U_t)
        distance = self.covar_dist(
            z1,
            z2,
            square_dist=True,
            diag=diag,
            **params,
        )
        return distance.mul(-0.5).exp()


class ALEBOGP(SingleTaskGP):
    """The GP for ALEBO.

    Uses the Mahalanobis kernel defined in ALEBOKernel, along with a
    ScaleKernel to add a kernel variance and a fitted constant mean.

    In non-batch mode, there is a single kernel that produces MVN predictions
    as usual for a GP.
    With b batches, each batch has its own set of kernel hyperparameters and
    each batch represents a sample from the hyperparameter posterior
    distribution. When making a prediction (with `__call__`), these samples are
    integrated over using moment matching. So, the predictions are an MVN as
    usual with the same shape as in non-batch mode.

    Args:
        B: (d x D) Projection matrix.
        train_X: (n x d) X training data.
        train_Y: (n x 1) Y training data.
        train_Yvar: (n x 1) Noise variances of each training Y.
    """

    def __init__(
        self, B: Tensor, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar, outcome_transform=None)
        self.covar_module = ScaleKernel(
            base_kernel=ALEBOKernel(B=B, batch_shape=self._aug_batch_shape),
            batch_shape=self._aug_batch_shape,
        )
        self.to(train_X)

    def __call__(self, x: Tensor) -> MultivariateNormal:
        """
        If model is non-batch, then just make a prediction. If model has
        multiple batches, then these are samples from the kernel hyperparameter
        posterior and we integrate over them with moment matching.

        The shape of the MVN that this outputs will be the same regardless of
        whether the model is batched or not.

        Args:
            x: Point to be predicted.

        Returns: MultivariateNormal distribution of prediction.
        """
        if len(self._aug_batch_shape) == 0:
            return super().__call__(x)
        # Else, approximately integrate over batches with moment matching.
        # Take X as (b) x q x d, and expand to (b) x ns x q x d
        if x.ndim > 3:
            raise ValueError("Don't know how to predict this shape")  # pragma: no cover
        x = x.unsqueeze(-3).expand(
            x.shape[:-2]
            + torch.Size([self._aug_batch_shape[0]])  # pyre-ignore
            + x.shape[-2:]
        )
        mvn_b = super().__call__(x)
        mu = mvn_b.mean.mean(dim=-2)
        C = (
            mvn_b.covariance_matrix.mean(dim=-3)
            + torch.matmul(mvn_b.mean.transpose(-2, -1), mvn_b.mean)
            / mvn_b.mean.shape[-2]
            - torch.matmul(mu.unsqueeze(-1), mu.unsqueeze(-2))
        )  # Law of Total Covariance
        mvn = MultivariateNormal(mu, C)
        return mvn

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        assert output_indices is None
        assert not observation_noise
        mvn = self(X)
        return GPyTorchPosterior(distribution=mvn)

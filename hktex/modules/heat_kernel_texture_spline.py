"""Anisotropic heat-kernel appearance on a B-spline surface.

Interface draft — signatures and docstrings only, no implementation.
See docs/spline_texture_design.md for the full design.

This is the (u,v)-domain analogue of
:class:`hktex.modules.heat_kernel_texture_knn.HeatKernelTextureKNN`. Each
kernel is an anisotropic Gaussian on the parameter domain whose covariance is
the inverse first fundamental form at the kernel centre, so the entire
discrete Laplacian eigen-decomposition and KNN heat propagation of HKTex is
replaced by a closed-form, control-point-differentiable evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hktex.utils.typing import *
from .heat_kernel_texture import HeatKernelTexture
from .spline_surface import SplineSurface

__all__ = ["HeatKernelTextureSpline"]


class HeatKernelTextureSpline(HeatKernelTexture):
    """Anisotropic heat-kernel appearance on a B-spline surface."""

    @dataclass
    class Config(HeatKernelTexture.Config):
        init_uv: str = "fps"               # "fps" | "uniform"
        use_metric_covariance: bool = True
        optimize_geometry: bool = False

    cfg: Config
    surface: SplineSurface

    # --- learnable kernel parameters ---
    _kernel_uv: nn.Parameter          # [G 2]
    _kernel_colours: nn.Parameter     # [G D]
    _angles: nn.Parameter             # [G]
    _anisotropies: nn.Parameter       # [G]
    _sigmas: nn.Parameter             # [G]

    def configure(self, surface: SplineSurface, **kwargs) -> None:
        """Store the spline surface and initialise kernel centres in (u,v).
        Reuses the activation/range-enforcement machinery of
        :meth:`HeatKernelTexture.configure`."""
        raise NotImplementedError

    # ---- kernel evaluation (core replacement) ----
    def kernel_weights(
        self, query_uv: Float[Tensor, "P 2"]
    ) -> Float[Tensor, "P G"]:
        r"""For each kernel k with centre c_k, anisotropy A_k and scale
        sigma_k:

            w_k(u,v) = exp( -1/(4 sigma_k^2)  d^T  g^{-1}(c_k) A_k^{-1} g^{-1}(c_k) d )

        with d = (u,v) - c_k and g(c_k) the first fundamental form at the
        centre. When ``use_metric_covariance`` is False, g^{-1} is replaced
        by the identity (isotropic-in-parameter ablation)."""
        raise NotImplementedError

    # ---- query interface ----
    def query(
        self, query_uv: Float[Tensor, "P 2"]
    ) -> tuple[Float[Tensor, "P D"], Float[Tensor, "P G"] | None]:
        """Evaluate appearance at P parameter points:
        colour(u,v) = mean_colour + sum_k w_k(u,v) * colour_k (normalised),
        optionally through ``out_net``. Returns (colour, contributions)."""
        raise NotImplementedError

    # ---- densification hooks ----
    def kernel_centres_uv(self) -> Float[Tensor, "G 2"]:
        raise NotImplementedError

    def add_kernels(self, new_uv: Float[Tensor, "G' 2"]) -> None:
        raise NotImplementedError

    def prune_kernels(self, keep_mask: Bool[Tensor, "G"]) -> None:
        raise NotImplementedError

    def resample_kernel_geometry(self) -> None:
        """Recompute cached g^{-1}(c_k) for every kernel centre after the
        control points have moved (Stage 2). O(G) closed-form evaluations."""
        raise NotImplementedError

    # ---- geometry coupling ----
    @property
    def surface_area(self) -> Float[Tensor, "1"]:
        r"""Integral of sqrt(det g) over [0,1]^2 by quadrature — area
        regulariser for Stage 2."""
        raise NotImplementedError

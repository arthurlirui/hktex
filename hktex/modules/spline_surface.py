"""Differentiable tensor-product B-spline surface carrier.

Interface draft — signatures and docstrings only, no implementation.
See docs/spline_texture_design.md for the full design.

This module plays the geometric-carrier role that
:class:`hktex.modules.mesh.Mesh` plays for HKTex, but the surface is a smooth
parametric B-spline whose first fundamental form is available in closed form
and differentiable in the control points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from hktex.utils.typing import *

__all__ = ["SplineSpec", "SplineSurface"]


@dataclass
class SplineSpec:
    """Immutable structure of a tensor-product B-spline surface.

    Control points and weights are *not* part of the spec: they are
    learnable parameters of :class:`SplineSurface`.
    """
    degree_u: int
    degree_v: int
    knots_u: Tensor
    knots_v: Tensor
    n_ctrl_u: int
    n_ctrl_v: int


class SplineSurface(nn.Module):
    """Differentiable tensor-product B-spline surface.

    All differential-geometry quantities (first fundamental form, metric
    inverse, normal, area element) are computed in closed form from the
    B-spline basis derivatives via the Diff-NURBS backend; nothing is
    discretised. ``to_mesh`` is the only place discretisation happens, and
    solely for rendering.
    """

    spec: SplineSpec
    control_points: nn.Parameter   # [Nu Nv 3]
    weights: nn.Parameter          # [Nu Nv]

    def __init__(
        self,
        control_points: Float[Tensor, "Nu Nv 3"],
        weights: Float[Tensor, "Nu Nv"],
        knots_u: Float[Tensor, "Ku"],
        knots_v: Float[Tensor, "Kv"],
        degree_u: int = 3,
        degree_v: int = 3,
        device: Optional[str] = None,
        optimize_geometry: bool = True,
    ) -> None:
        raise NotImplementedError

    # ---- core surface evaluation ----
    def evaluate(self, uv: Float[Tensor, "P 2"]) -> Float[Tensor, "P 3"]:
        """S(u,v). Differentiable in ``control_points`` and ``weights``."""
        raise NotImplementedError

    def derivatives(
        self, uv: Float[Tensor, "P 2"]
    ) -> tuple[Float[Tensor, "P 3"], Float[Tensor, "P 3"], Float[Tensor, "P 3"]]:
        """Return (S, dS/du, dS/dv), all differentiable."""
        raise NotImplementedError

    # ---- differential geometry ----
    def first_fundamental_form(
        self, uv: Float[Tensor, "P 2"]
    ) -> Float[Tensor, "P 2 2"]:
        r"""g(u,v) = [[E, F], [F, G]] with
        E = dS/du . dS/du, F = dS/du . dS/dv, G = dS/dv . dS/dv.
        Closed-form; differentiable in control points."""
        raise NotImplementedError

    def metric_inverse(
        self, uv: Float[Tensor, "P 2"]
    ) -> Float[Tensor, "P 2 2"]:
        r"""g^{-1}(u,v) — the base covariance of each heat kernel."""
        raise NotImplementedError

    def normal(self, uv: Float[Tensor, "P 2"]) -> Float[Tensor, "P 3"]:
        """Unit normal n = normalize(dS/du x dS/dv)."""
        raise NotImplementedError

    def area_element(self, uv: Float[Tensor, "P 2"]) -> Float[Tensor, "P"]:
        """sqrt(det g). For area regularisation and kernel normalisation."""
        raise NotImplementedError

    # ---- rendering bridge ----
    def to_mesh(
        self, resolution_u: int, resolution_v: int
    ) -> tuple[Float[Tensor, "V 3"], Int[Tensor, "F 3"], Float[Tensor, "V 2"]]:
        """Adaptively subdivide S into a dense triangular mesh for the
        existing Mitsuba ray renderer. Each vertex carries its (u,v) so a
        ray hit recovers the surface parameter. Gradients flow back through
        the subdivision matrix."""
        raise NotImplementedError

    # ---- optimisation toggles ----
    def freeze_geometry(self) -> None:
        raise NotImplementedError

    def unfreeze_geometry(self) -> None:
        raise NotImplementedError

    @property
    def geometry_optimizable(self) -> bool:
        raise NotImplementedError

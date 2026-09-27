"""Joint differentiable B-spline geometry + heat-kernel appearance model.

Interface draft — signatures and docstrings only, no implementation.
See docs/spline_texture_design.md for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

import hktex
from hktex.modules.base import TextureModel
from hktex.modules.spline_surface import SplineSurface
from hktex.modules.heat_kernel_texture_spline import HeatKernelTextureSpline
from hktex.utils.typing import *

__all__ = ["SplineTextureModel"]


@hktex.register("modules.spline-texture-model")
class SplineTextureModel(TextureModel):
    """Joint differentiable B-spline geometry + heat-kernel appearance model.

    Stage 1: ``optimize_geometry=False`` — control points frozen, only
             kernel parameters and colours are learned.
    Stage 2: ``optimize_geometry=True`` — control points and kernels
             optimised jointly from multi-view images with area/curvature
             regularisation and an appearance-geometry decoupling schedule.
    """

    @dataclass
    class Config(TextureModel.Config):
        surface: dict = field(default_factory=dict)
        texture: dict = field(default_factory=dict)
        optimize_geometry: bool = False
        point_batching: Optional[int] = None

    cfg: Config
    surface: SplineSurface
    texture: HeatKernelTextureSpline

    def configure(self, **kwargs) -> None:
        raise NotImplementedError

    # ---- TextureModel interface ----
    def requires_scene_bounds(self) -> bool:
        return False

    def requires_face_ids(self) -> bool:
        # The spline model is queried by (u,v), not by mesh face.
        return False

    def forward(
        self,
        uv: Float[Tensor, "P 2"],
        **kwargs,
    ) -> Float[Tensor, "P out_dim"]:
        """Return appearance at P parameter points."""
        raise NotImplementedError

    def post_optimizer_step(self) -> None:
        """Stage 2: clamp parameters and resample cached kernel geometry
        after the control-point update."""
        raise NotImplementedError

    # ---- regularisation exposed to the trainer ----
    def geometry_regulariser(self) -> dict:
        """Return {'area': ..., 'curvature': ...} for the trainer to combine
        with the rendering loss. Empty in Stage 1."""
        raise NotImplementedError

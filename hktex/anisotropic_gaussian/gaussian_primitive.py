"""Per-Gaussian primitive model: parameters, tangent frame, material fields.

This is the data container for the anisotropic-Gaussian splat. It holds:

- the usual 3DGS attributes: center ``xyz``, scaling, rotation (quaternion),
  opacity, base color;
- the *tangent frame* (u, v, n0) derived from scaling+rotation, which makes
  each primitive an oriented disk -- this is the 2DGS / surfel formulation
  that DeferredGS and R3DG both use, and that HKTex's geodesic Gaussians
  generalize to surfaces;
- the anisotropic BRDF parameters: ``metallic``, ``alpha_t``, ``alpha_b`` --
  the last two are the ASG roughnesses along the tangent and bitangent;
- the tangent-space normal perturbation field ``(d_x, d_y)`` from
  contribution 1.

The model is intentionally framework-light: it stores tensors and exposes
accessors, but does not depend on any specific rasterizer. The deferred
renderer in :mod:`hktex.anisotropic_gaussian.deferred_splat` consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .asg_brdf import _safe_normalize
from .normal_perturb import perturb_normal_tangent_space


@dataclass
class PrimitiveAttributes:
    """Bundle of per-primitive attribute tensors, all (N, K)."""

    xyz: torch.Tensor            # (N, 3)  center
    scaling: torch.Tensor        # (N, 3)  log-space, exp() to get radii
    rotation: torch.Tensor       # (N, 4)  quaternion (w, x, y, z)
    opacity: torch.Tensor        # (N, 1)  logit
    base_color: torch.Tensor     # (N, 3)  linear RGB, sigmoid
    metallic: torch.Tensor       # (N, 1)  sigmoid
    alpha_t: torch.Tensor        # (N, 1)  ASG roughness along tangent, softplus
    alpha_b: torch.Tensor        # (N, 1)  ASG roughness along bitangent, softplus
    normal_perturb: torch.Tensor  # (N, 2)  tangent-space offset field, raw


def quaternion_to_tangent_frame(
    scaling: torch.Tensor, rotation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a surfel's tangent frame (u, v, n0) from scaling and rotation.

    The disk lies in the local xy-plane with radii (s_x, s_y); the quaternion
    rotates it into world space. The base normal is ``n0 = normalize(u x v)``.
    This is the same construction 2DGS/DeferredGS use to turn a Gaussian
    primitive into an oriented surfel.

    Parameters
    ----------
    scaling : (N, 3)
        Radii (already exp-activated). Only the first two components are
        used for the disk; the third is the (unused here) thickness.
    rotation : (N, 4)
        Unit quaternions in (w, x, y, z) convention.

    Returns
    -------
    u, v, n0 : each (N, 3)
    """
    # Build rotation matrix from quaternion (w, x, y, z).
    w, x, y, z = rotation.unbind(-1)
    # Row-major R such that R @ local = world. Standard quaternion-to-matrix.
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*rotation.shape[:-1], 3, 3)  # (N, 3, 3)

    # Local tangent axes, scaled by the disk radii.
    sx = scaling[..., 0:1]
    sy = scaling[..., 1:2]
    u = (R[..., :, 0] * sx)  # (N, 3)
    v = (R[..., :, 1] * sy)
    n0 = _safe_normalize(torch.linalg.cross(u, v, dim=-1))
    return u, v, n0


class AnisotropicGaussianModel(nn.Module):
    """Learnable set of anisotropic Gaussian surfels.

    Attributes are stored as ``nn.Parameter`` so the model can be optimized
    end-to-end with the rest of the pipeline. The activations (exp for
    scaling, sigmoid for opacity/color/metallic, softplus for roughness)
    are applied in the accessors so the raw parameters live in an
    unconstrained space for optimization stability -- the same convention
    as 3DGS/DeferredGS/R3DG.
    """

    def __init__(self, n_primitives: int, init_alpha: float = 0.3):
        super().__init__()
        # Initialize attributes. In practice these would be seeded from an
        # SfM point cloud (as 3DGS does); here we provide a random init so
        # the module is runnable standalone for testing the integration.
        self.n_primitives = n_primitives

        self.xyz = nn.Parameter(torch.randn(n_primitives, 3) * 0.5)
        # Log-space scaling; init to a small disk.
        self.scaling = nn.Parameter(torch.zeros(n_primitives, 3))
        # Identity quaternion (w=1).
        q = torch.zeros(n_primitives, 4)
        q[:, 0] = 1.0
        self.rotation = nn.Parameter(q)
        self.opacity = nn.Parameter(torch.zeros(n_primitives, 1))
        self.base_color = nn.Parameter(torch.zeros(n_primitives, 3))
        self.metallic = nn.Parameter(torch.full((n_primitives, 1), -2.0))  # ~0.12
        # Roughness init: log(init_alpha) so softplus(log(x)) ~ x.
        log_a = torch.log(torch.tensor(init_alpha)).item()
        self.alpha_t = nn.Parameter(torch.full((n_primitives, 1), log_a))
        self.alpha_b = nn.Parameter(torch.full((n_primitives, 1), log_a))
        # Normal perturbation starts at zero -> no perturbation initially,
        # so the model starts as a flat-disk splat and learns detail.
        self.normal_perturb = nn.Parameter(torch.zeros(n_primitives, 2))

    # --- accessors with activations -----------------------------------------
    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.scaling)

    def get_rotation(self) -> torch.Tensor:
        return self.rotation / (self.rotation.norm(dim=-1, keepdim=True) + 1e-8)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity)

    def get_base_color(self) -> torch.Tensor:
        return torch.sigmoid(self.base_color)

    def get_metallic(self) -> torch.Tensor:
        return torch.sigmoid(self.metallic)

    def get_alpha_t(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_t) + 1e-3

    def get_alpha_b(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_b) + 1e-3

    def get_normal_perturb(self) -> torch.Tensor:
        return self.normal_perturb

    def get_tangent_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the *geometric* tangent frame (u, v, n0)."""
        return quaternion_to_tangent_frame(self.get_scaling(), self.get_rotation())

    def get_shading_normal(self) -> torch.Tensor:
        """Return the *perturbed* shading normal (contribution 1)."""
        u, v, n0 = self.get_tangent_frame()
        return perturb_normal_tangent_space(
            n0, u, v, self.get_normal_perturb()
        )

    def get_attributes(self) -> PrimitiveAttributes:
        return PrimitiveAttributes(
            xyz=self.get_xyz(),
            scaling=self.get_scaling(),
            rotation=self.get_rotation(),
            opacity=self.get_opacity(),
            base_color=self.get_base_color(),
            metallic=self.get_metallic(),
            alpha_t=self.get_alpha_t(),
            alpha_b=self.get_alpha_b(),
            normal_perturb=self.get_normal_perturb(),
        )

    def to_parameter_dict(self) -> dict[str, torch.Tensor]:
        """Return raw (unactivated) parameters for an optimizer."""
        return {
            "xyz": self.xyz,
            "scaling": self.scaling,
            "rotation": self.rotation,
            "opacity": self.opacity,
            "base_color": self.base_color,
            "metallic": self.metallic,
            "alpha_t": self.alpha_t,
            "alpha_b": self.alpha_b,
            "normal_perturb": self.normal_perturb,
        }

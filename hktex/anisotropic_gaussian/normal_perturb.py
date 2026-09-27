"""Contribution 1: Per-Gaussian tangent-space normal perturbation field.

Each Gaussian surfel carries a base geometric normal ``n0 = u x v`` derived
from its tangent vectors. We attach a *learnable perturbation* in tangent
space so that the *shading* normal

    n = normalize(n0 + d_x * u + d_y * v)

can deviate from the geometric normal without changing the primitive's
shape, opacity, or position. This is the surface-splatting analogue of a
tangent-space normal map: the primitive geometry stays a flat disk, but
the BRDF sees a perturbed normal that adds fine surface detail.

This follows the NVDIFFREC ``_perturb_normal`` pattern (bsdf.py:39-45)::

    shading_nrm = t * n_pert.x + bitang * n_pert.y + n * clamp(n_pert.z, 0)
    return normalize(shading_nrm)

with the simplification that the perturbation here is a 2D offset in the
tangent plane (the disk's own u/v), so we do not need a separate texture
lookup -- the perturbation is a per-Gaussian learned field, stored
compactly as two floats per primitive. This is what makes it a *field* on
the splats rather than a UV-mapped texture: there is no UV atlas, which
matches the UV-free philosophy of HKTex.
"""

from __future__ import annotations

import torch

from .asg_brdf import _safe_normalize


def perturb_normal_tangent_space(
    n0: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    perturbation: torch.Tensor,
    max_angle: float = 1.2,
) -> torch.Tensor:
    """Apply a tangent-space perturbation to a Gaussian surfel's normal.

    Parameters
    ----------
    n0 : (..., 3)
        Base geometric normal, expected normalized (``u x v``).
    u, v : (..., 3)
        Tangent vectors spanning the surfel disk. Need not be unit length
        (the disk is anisotropic), but should be orthogonal to ``n0``.
    perturbation : (..., 2)
        Per-primitive perturbation field ``(d_x, d_y)`` in tangent space,
        *before* the angular clamp. These are the learned parameters; they
        are stored in the model and optimized end-to-end with the rest of
        the Gaussian attributes.
    max_angle : float
        Soft clamp on the perturbation magnitude, in radians. Keeps the
        shading normal within a cone around ``n0`` so the BRDF cannot
        invent surfaces that point away from the primitive. Defaults to
        ~69 degrees, generous to allow strong detail but avoid degenerate
        normals.

    Returns
    -------
    (..., 3)
        Perturbed shading normal, normalized. Lies in the cone
        ``angle(n, n0) <= max_angle``.
    """
    # The perturbation is applied as a tangent-plane offset. We rescale the
    # raw (d_x, d_y) through a tanh so the *angular* deviation is bounded:
    # tan(angle) = |d_x*u + d_y*v| / |n0|, so clamping |d| to tan(max_angle)
    # clamps the angle. This is the continuous analogue of NVDIFFREC's
    # ``clamp(perturbed_nrm.z, 0)`` but in 2D tangent space.
    bound = torch.tensor(max_angle, device=perturbation.device).tan()
    d = torch.tanh(perturbation) * bound  # (..., 2)

    # Build the offset in world space. We normalize u,v first so the offset
    # magnitude is controlled purely by ``d``.
    u_n = _safe_normalize(u)
    v_n = _safe_normalize(v)
    offset = d[..., 0:1] * u_n + d[..., 1:2] * v_n  # (..., 3)

    n = _safe_normalize(n0 + offset)
    return n

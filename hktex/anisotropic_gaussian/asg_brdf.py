"""Contribution 2: ASG anisotropic microfacet BRDF for Gaussian splatting.

The normal distribution function is an Anisotropic Spherical Gaussian (ASG)
whose lobe axis is the primitive normal ``n`` and whose two tangent
directions ``t`` (tangent) and ``b`` (bitangent) carry independent sharpness
parameters ``alpha_t`` and ``alpha_b``::

    D(h) = exp( -((h.t)^2 / alpha_t^2 + (h.b)^2 / alpha_b^2) )

This is the anisotropic generalization of the isotropic SG NDF
``D(h) = (1/(alpha^2 pi)) * exp((2/alpha^2)(h.n - 1))`` used in
R3DG's CUDA render equation (``render_equation.cu`` line 154). Keeping the
NDF in this exponential form makes it a drop-in replacement: the rest of
the Cook-Torrance terms (Schlick Fresnel, Smith joint visibility) are
unchanged, only ``D`` becomes anisotropic and the visibility term takes
two roughnesses.

The ASG NDF is *not* normalized to integrate to 1 over the hemisphere
(the closed-form normalization of an anisotropic exponential on the sphere
is messy); following common practice in real-time rendering (Frostbite,
Unreal's anisotropic BRDF) we apply a discrete normalization constant
``1 / (pi * alpha_t * alpha_b)`` so that the BRDF is energy-conserving
for Lambertian albedo and the specular lobe energy is bounded. This is
the same convention DeferredGS/NVDIFFREC use for their GGX LUT path and
matches the ``amp = 1/(r^2 pi)`` prefactor R3DG applies to its SG NDF.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

# Numerical floor for roughness to keep the exponent finite and the
# normalization constant bounded. Matches R3DG's ``fmaxf(rough*rough, 1e-7)``.
_ROUGHNESS_EPS = 1e-4


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def asg_ndf(
    h: torch.Tensor,
    n: torch.Tensor,
    t: torch.Tensor,
    b: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_b: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic Spherical Gaussian normal distribution function.

    Parameters
    ----------
    h : (..., 3)
        Half-vector, expected normalized.
    n, t, b : (..., 3)
        Primitive shading normal, tangent, bitangent. Must form an
        orthonormal frame (caller is responsible; see
        :func:`hktex.anisotropic_gaussian.gaussian_primitive.AnisotropicGaussianModel.get_tangent_frame`).
    alpha_t, alpha_b : (..., 1)
        Roughness along the tangent and bitangent directions. These are the
        *anisotropic* generalization of R3DG's single ``rough`` parameter;
        smaller = sharper, mirror-like; larger = broader, diffuse-like.

    Returns
    -------
    (..., 1)
        NDF value ``D(h)``. Bounded to be non-negative; the half-vector is
        clamped to the upper hemisphere so back-facing half-vectors do not
        produce negative exponentials.
    """
    # Clamp the half-vector into the upper hemisphere of the shading normal.
    # Without this, grazing back-side half-vectors produce ``h.n < 0`` which
    # drives the isotropic fallback negative; ASG uses (h.t)^2 and (h.b)^2
    # which are always positive, but we still want D -> 0 when h is below
    # the surface, so we gate on h.n.
    h_dot_n = (h * n).sum(dim=-1, keepdim=True).clamp_min(0.0)
    h_dot_t = (h * t).sum(dim=-1, keepdim=True)
    h_dot_b = (h * b).sum(dim=-1, keepdim=True)

    a_t = alpha_t.clamp_min(_ROUGHNESS_EPS)
    a_b = alpha_b.clamp_min(_ROUGHNESS_EPS)

    # Exponential anisotropic lobe. The ``h.n`` gate zeroes back-facing
    # half-vectors; the squared tangent projections give the anisotropy.
    exponent = -((h_dot_t * h_dot_t) / (a_t * a_t) + (h_dot_b * h_dot_b) / (a_b * a_b))
    # Re-center the lobe on the normal: when h == n, both squared projections
    # vanish and exponent == 0, so D == normalization. As h tilts away from
    # n along t or b, the corresponding squared projection grows and D
    # decays anisotropically. This is exactly the ASG form used in
    # Spec-Gaussian's RenderingEquationEncoding (spec_utils.py:80-87),
    # recast as a physically shaded NDF rather than a learned appearance
    # field.
    D = torch.exp(exponent) / (math.pi * a_t * a_b)
    return D * h_dot_n  # gate to upper hemisphere


def schlick_fresnel(h_dot_o: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """Schlick Fresnel, identical to R3DG (render_equation.cu:158).

    ``f0`` is the reflectance at normal incidence; for metals it is the
    base color, for dielectrics it is 0.04.
    """
    one_minus = (1.0 - h_dot_o).clamp(0.0, 1.0)
    return f0 + (1.0 - f0) * (one_minus ** 5)


def smith_visibility_aniso(
    n_dot_i: torch.Tensor,
    n_dot_o: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_b: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic Smith joint visibility term V(n,i,o).

    Generalizes R3DG's isotropic ``V`` (render_equation.cu:161-162), which
    uses ``r2 = (1+rough)^2 / 8`` and the Smith-GGX height-correlated form.
    We use the anisotropic Smith lambda sum with the ASG's
    ``alpha_t / alpha_b`` roughnesses, projected onto the tangent frame via
    the half-vector's tangent-space components. Because we do not have the
    half-vector here, we use the conservative isotropic fallback with the
    geometric mean roughness ``alpha_g = sqrt(alpha_t * alpha_b)``, which is
    the standard anisotropic-to-isotropic reduction used when the lighting
    is integrated over the full hemisphere (Heitz 2014, Sec. 5.3).
    """
    a_g = (alpha_t * alpha_b).clamp_min(_ROUGHNESS_EPS).sqrt()
    # Smith-GGX height-correlated visibility, isotropic form with a_g.
    # This matches R3DG's ``r2 = (1+rough)^2/8`` family but uses the
    # cleaner Karis/Heitz formulation.
    a_g2 = a_g * a_g
    vis_i = n_dot_i / (n_dot_i + (a_g2 + (1.0 - a_g2) * n_dot_i).sqrt() + 1e-8)
    vis_o = n_dot_o / (n_dot_o + (a_g2 + (1.0 - a_g2) * n_dot_o).sqrt() + 1e-8)
    return vis_i * vis_o


def asg_brdf(
    wo: torch.Tensor,
    n: torch.Tensor,
    t: torch.Tensor,
    b: torch.Tensor,
    wi: torch.Tensor,
    base_color: torch.Tensor,
    metallic: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full Cook-Torrance BRDF with the ASG NDF.

    Returns ``(f_d, f_s)`` -- the diffuse and specular reflectance,
    *not yet* multiplied by the incident radiance or the cosine foreshortening.
    The caller (the deferred shader) is responsible for the lighting integral,
    matching the structure of R3DG's render equation where ``rgb_d += f_d *
    transport`` and ``rgb_s += f_s * transport``.

    Parameters
    ----------
    wo, wi : (..., 3)
        Outgoing (view) and incident (light) directions, pointing *away*
        from the surface.
    n, t, b : (..., 3)
        Shading normal / tangent / bitangent. ``n`` should already be the
        *perturbed* normal from
        :func:`hktex.anisotropic_gaussian.normal_perturb.perturb_normal_tangent_space`
        if contribution 1 is active.
    base_color, metallic, alpha_t, alpha_b : (..., 3|1|1|1)
        Per-primitive material parameters.

    Returns
    -------
    f_d : (..., 3)  -- Lambertian diffuse, ``(1-metal)*base/pi``.
    f_s : (..., 3)  -- Specular, ``D*F*V`` Cook-Torrance.
    """
    n = _safe_normalize(n)
    wo = _safe_normalize(wo)
    wi = _safe_normalize(wi)
    h = _safe_normalize(wo + wi)

    n_dot_i = (n * wi).sum(dim=-1, keepdim=True).clamp_min(1e-4)
    n_dot_o = (n * wo).sum(dim=-1, keepdim=True).clamp_min(1e-4)
    h_dot_o = (h * wo).sum(dim=-1, keepdim=True).clamp_min(0.0)

    # --- Diffuse (R3DG:148) ---------------------------------------------------
    f_d = (1.0 - metallic) * base_color / math.pi

    # --- Specular: D (ASG) * F (Schlick) * V (Smith) -------------------------
    D = asg_ndf(h, n, t, b, alpha_t, alpha_b)
    f0 = 0.04 * (1.0 - metallic) + base_color * metallic
    F = schlick_fresnel(h_dot_o, f0)
    V = smith_visibility_aniso(n_dot_i, n_dot_o, alpha_t, alpha_b)
    f_s = D * F * V

    # Zero out back-facing configurations to avoid energy leaks.
    valid = ((n * wi).sum(dim=-1, keepdim=True) > 0.0).float() * \
            ((n * wo).sum(dim=-1, keepdim=True) > 0.0).float()
    return f_d * valid, f_s * valid

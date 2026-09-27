"""Anisotropic Gaussian Splatting with Normal-Mapped Microfacet Shading.

Integration of three contributions targeting SIGGRAPH/TOG:

1. Per-Gaussian tangent-space normal perturbation field
   (:mod:`hktex.anisotropic_gaussian.normal_perturb`) -- adds fine surface
   detail to each Gaussian surfel without changing its primitive geometry,
   following the tangent-space normal-mapping pattern of NVDIFFREC.

2. Anisotropic microfacet shading model for Gaussian splatting
   (:mod:`hktex.anisotropic_gaussian.asg_brdf`) -- an ASG (Anisotropic
   Spherical Gaussian) normal distribution function aligned to the
   primitive tangent frame, replacing the isotropic SG NDF used in
   R3DG's CUDA render equation.

3. Deferred splatting pipeline
   (:mod:`hktex.anisotropic_gaussian.deferred_splat`) -- a two-pass
   rasterizer/shader that first accumulates a G-buffer of per-Gaussian
   material parameters and then evaluates the ASG BRDF per pixel against
   an environment map, mirroring DeferredGS's G-buffer architecture.

The module is self-contained pure PyTorch so it can be imported and run
without compiling any CUDA extension; see :mod:`hktex.anisotropic_gaussian.render`
for the top-level entry point and :mod:`hktex.anisotropic_gaussian.train`
for a minimal training-loop demonstration.
"""

from .gaussian_primitive import AnisotropicGaussianModel
from .asg_brdf import asg_ndf, asg_brdf, schlick_fresnel, smith_visibility_aniso
from .normal_perturb import perturb_normal_tangent_space
from .deferred_splat import DeferredSplatRenderer, GBuffer, Camera
from .render import render

__all__ = [
    "AnisotropicGaussianModel",
    "DeferredSplatRenderer",
    "GBuffer",
    "Camera",
    "asg_ndf",
    "asg_brdf",
    "schlick_fresnel",
    "smith_visibility_aniso",
    "perturb_normal_tangent_space",
    "render",
]

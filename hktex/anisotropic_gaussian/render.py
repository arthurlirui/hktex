"""Top-level render entry point for the anisotropic-Gaussian integration.

This ties together the three contributions into a single callable:

    rgb, gbuffer = render(model, camera, renderer)

It is the public API used by :mod:`hktex.anisotropic_gaussian.train` and by
any downstream viewer. The function is deliberately thin -- all the work is
in the :class:`~hktex.anisotropic_gaussian.deferred_splat.DeferredSplatRenderer`,
which already combines:

- contribution 1 (normal perturbation) via
  :meth:`AnisotropicGaussianModel.get_shading_normal`,
- contribution 2 (ASG BRDF) via :func:`hktex.anisotropic_gaussian.asg_brdf.asg_brdf`,
- contribution 3 (deferred splatting) via the two-pass
  :meth:`DeferredSplatRenderer.forward`.

Keeping this layer thin matches the structure of the cloned references:
3DGS/DeferredGS/R3DG all expose a top-level ``render(camera, pc, pipe, bg)``
that delegates to the Gaussian rasterizer and the BRDF shader.
"""

from __future__ import annotations

from typing import Optional

import torch

from .deferred_splat import Camera, DeferredSplatRenderer, GBuffer
from .gaussian_primitive import AnisotropicGaussianModel


def render(
    model: AnisotropicGaussianModel,
    camera: Camera,
    renderer: Optional[DeferredSplatRenderer] = None,
    *,
    image_height: int = 256,
    image_width: int = 256,
    num_env_sg: int = 32,
    num_light_samples: int = 64,
) -> tuple[torch.Tensor, GBuffer]:
    """Render the anisotropic-Gaussian scene from one camera.

    Parameters
    ----------
    model : AnisotropicGaussianModel
        The per-Gaussian primitive model carrying xyz, tangent frame,
        anisotropic roughnesses, and the normal-perturbation field.
    camera : Camera
        Pinhole camera. Must be on the same device as ``model``.
    renderer : DeferredSplatRenderer, optional
        If provided, reuse it (and its learnable envmap); otherwise build
        one with the given resolution/sample counts. Reusing is the common
        case during training so the envmap persists across views.
    image_height, image_width : int
        Resolution, used only if ``renderer`` is None.
    num_env_sg : int
        Number of SG lobes in the envmap, used only if ``renderer`` is None.
    num_light_samples : int
        Monte Carlo samples for the per-pixel BRDF integral, used only if
        ``renderer`` is None.

    Returns
    -------
    rgb : (H, W, 3) torch.Tensor
        Shaded image in linear RGB.
    gbuffer : GBuffer
        The deferred G-buffer (pass-1 output), exposed for losses,
        visualization, and normal/roughness regularization.
    """
    if renderer is None:
        renderer = DeferredSplatRenderer(
            image_height=image_height,
            image_width=image_width,
            num_env_sg=num_env_sg,
            num_light_samples=num_light_samples,
        ).to(model.xyz.device)
    return renderer(model, camera)

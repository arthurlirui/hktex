"""Contribution 3: Deferred splatting pipeline.

Two-pass renderer:

Pass 1 -- G-buffer splatting
    For each Gaussian surfel, splat its material attributes into screen-space
    G-buffer channels using alpha compositing (front-to-back, over operator).
    The G-buffer holds, per pixel:

    - ``base_color``    (3)
    - ``shading_normal`` (3)   -- *already perturbed* by contribution 1
    - ``tangent``       (3)    -- u, for the ASG NDF
    - ``bitangent``     (3)    -- v, for the ASG NDF
    - ``metallic``      (1)
    - ``alpha_t``       (1)
    - ``alpha_b``       (1)
    - ``view_pos``      (3)    -- world-space hit position for the view dir
    - ``alpha``         (1)    -- accumulated opacity, for the background mask

    This mirrors DeferredGS's G-buffer (gaussian_renderer/__init__.py:114-181),
    where the rasterizer emits normal/depth/diffuse/specular/roughness and
    the Python side then calls ``brdf_mlp.shade(...)`` per pixel. Here the
    shade step is the ASG BRDF instead of an MLP.

    Because we do not ship a CUDA rasterizer, the splatting pass is a pure
    PyTorch reference implementation: each primitive projects to an ellipse
    and accumulates into the G-buffer via tensor indexing. This is slow but
    correct, and keeps the module importable without a CUDA toolchain. A
    drop-in CUDA rasterizer (e.g. DeferredGS's ``diff_surfel_rasterization``)
    can replace the ``_splat_reference`` call without touching the shader.

Pass 2 -- Per-pixel ASG shading
    For each lit pixel, evaluate the ASG BRDF against an environment map
    represented as a set of Spherical Gaussians (the same representation
    Spec-Gaussian's ``SGEnvmap`` uses, spec_utils.py:92-109). The integral

        L_o = integral_hemisphere f_r(wo, wi) L_i(wi) cos(theta_i) dwi

    is approximated by Monte Carlo over a fixed sample set, exactly as
    R3DG does in render_equation.cu:90-179 (Fibonacci spiral on the
    hemisphere, SH-free since we have an explicit envmap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .asg_brdf import asg_brdf, _safe_normalize
from .gaussian_primitive import AnisotropicGaussianModel, PrimitiveAttributes


@dataclass
class GBuffer:
    """Screen-space G-buffer produced by pass 1, consumed by pass 2."""

    base_color: torch.Tensor     # (H, W, 3)
    shading_normal: torch.Tensor  # (H, W, 3)
    tangent: torch.Tensor        # (H, W, 3)
    bitangent: torch.Tensor      # (H, W, 3)
    metallic: torch.Tensor       # (H, W, 1)
    alpha_t: torch.Tensor        # (H, W, 1)
    alpha_b: torch.Tensor        # (H, W, 1)
    view_pos: torch.Tensor       # (H, W, 3)  world-space hit position
    alpha: torch.Tensor          # (H, W, 1)  accumulated opacity

    @property
    def device(self) -> torch.device:
        return self.base_color.device

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "base_color": self.base_color,
            "shading_normal": self.shading_normal,
            "tangent": self.tangent,
            "bitangent": self.bitangent,
            "metallic": self.metallic,
            "alpha_t": self.alpha_t,
            "alpha_b": self.alpha_b,
            "view_pos": self.view_pos,
            "alpha": self.alpha,
        }


@dataclass
class Camera:
    """Minimal pinhole camera for the reference renderer."""

    world_view_transform: torch.Tensor  # (4, 4) world -> view
    full_proj_transform: torch.Tensor   # (4, 4) world -> clip
    camera_center: torch.Tensor         # (3,)    world-space eye
    image_height: int
    image_width: int
    znear: float = 0.01
    zfar: float = 100.0

    @property
    def device(self) -> torch.device:
        return self.world_view_transform.device

    @property
    def FoVx(self) -> float:
        # Infer from projection matrix: FoVx = 2 * atan(1 / P[0,0]).
        return 2.0 * math.atan(1.0 / float(self.full_proj_transform[0, 0]))

    @property
    def FoVy(self) -> float:
        return 2.0 * math.atan(1.0 / float(self.full_proj_transform[1, 1]))


class SGEnvironmentMap(nn.Module):
    """Spherical-Gaussian environment map for incident radiance.

    Identical representation to Spec-Gaussian's ``SGEnvmap``: ``numLgtSGs``
    lobes, each with axis (3), sharpness lambda (1), and amplitude (3).
    The forward returns radiance ``L_i(wi)`` for any world-space direction.
    """

    def __init__(self, num_sg: int = 32, init_scale: float = 0.2):
        super().__init__()
        self.num_sg = num_sg
        # Lobe axis, sharpness, amplitude stacked as (num_sg, 7).
        self.lgt_sg = nn.Parameter(torch.randn(num_sg, 7) * init_scale)
        # Bias the sharpness to a reasonable value and give the lobes a small
        # nonzero amplitude so the envmap starts with a dim but nonzero
        # incident-light field. Zero amplitude would zero the entire shading
        # pass and block gradients to the BRDF and material parameters until
        # the optimizer moves the amplitude away from zero, which wastes early
        # training steps.
        with torch.no_grad():
            self.lgt_sg.data[:, 3:4] *= 50.0  # sharpness
            self.lgt_sg.data[:, 4:7] = 0.1    # amplitude: dim gray sky

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        """Evaluate the envmap at directions ``dirs``.

        Parameters
        ----------
        dirs : (..., 3) normalized world-space directions.

        Returns
        -------
        (..., 3) incident radiance.
        """
        lobes = self.lgt_sg[:, :3]
        lobes = lobes / (lobes.norm(dim=-1, keepdim=True) + 1e-7)
        lambdas = self.lgt_sg[:, 3:4].abs()
        mus = self.lgt_sg[:, 4:7].abs()

        # (..., num_sg, 1)
        dots = (dirs[..., None, :] * lobes).sum(dim=-1, keepdim=True)
        radiance = mus[None] * torch.exp(lambdas[None] * (dots - 1.0))
        return radiance.sum(dim=-2)  # (..., 3)


def _project_to_screen(
    xyz: torch.Tensor, camera: Camera
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-space centers to (screen_x, screen_y) in pixels and depth.

    Returns ``(screen_xy, depth)`` where ``screen_xy`` is (N, 2) in pixel
    coordinates and ``depth`` is (N, 1) view-space depth.
    """
    # World -> clip.
    homog = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # (N, 4)
    clip = homog @ camera.full_proj_transform.T  # (N, 4)
    w = clip[..., 3:4].clamp_min(1e-6)
    ndc = clip[..., :3] / w  # (N, 3)
    # NDC -> pixels.
    H, W = camera.image_height, camera.image_width
    screen_x = (ndc[..., 0:1] * 0.5 + 0.5) * W
    screen_y = (1.0 - (ndc[..., 1:2] * 0.5 + 0.5)) * H  # flip y
    # View-space depth for ordering.
    view = homog @ camera.world_view_transform.T
    depth = view[..., 2:3]
    return torch.cat([screen_x, screen_y], dim=-1), depth


class DeferredSplatRenderer(nn.Module):
    """Deferred splatting renderer combining all three contributions.

    Parameters
    ----------
    image_height, image_width : int
        Output resolution.
    num_env_sg : int
        Number of SG lobes in the environment map.
    num_light_samples : int
        Monte Carlo samples for the per-pixel BRDF integral.
    splat_radius_pixels : float
        Reference-rasterizer-only: the screen-space radius of each splat,
        in pixels. A real CUDA rasterizer would derive this from the
        projected ellipse; here we use a constant for simplicity.
    """

    def __init__(
        self,
        image_height: int = 256,
        image_width: int = 256,
        num_env_sg: int = 32,
        num_light_samples: int = 64,
        splat_radius_pixels: float = 3.0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.num_light_samples = num_light_samples
        self.splat_radius_pixels = splat_radius_pixels
        self.envmap = SGEnvironmentMap(num_sg=num_env_sg)
        # Precompute Fibonacci-spiral hemisphere samples. These are fixed
        # (not learned) -- they define the integration directions. The
        # envmap provides L_i at each.
        self.register_buffer(
            "_hemisphere_samples", self._make_hemisphere_samples(num_light_samples), persistent=False
        )

    @staticmethod
    def _make_hemisphere_samples(n: int) -> torch.Tensor:
        """Fibonacci spiral on the upper hemisphere, uniform in solid angle.

        Matches R3DG (render_equation.cu:90-98), which uses the same spiral
        for its importance samples. Returns (n, 3) normalized directions in
        a local frame where +z is up.
        """
        golden = math.pi * (3.0 - math.sqrt(5.0))
        idx = torch.arange(n, dtype=torch.float32)
        z = 1.0 - 2.0 * idx / (2.0 * n - 1.0)
        z = z.clamp(0.0, 1.0)  # upper hemisphere only
        r = (1.0 - z * z).sqrt()
        theta = golden * idx
        x = torch.cos(theta) * r
        y = torch.sin(theta) * r
        return torch.stack([x, y, z], dim=-1)  # (n, 3)

    # --- Pass 1: G-buffer splatting -----------------------------------------
    def _splat_reference(
        self, model: AnisotropicGaussianModel, camera: Camera
    ) -> GBuffer:
        """Reference pure-PyTorch G-buffer splatting.

        This is a teaching implementation. Each primitive is splatted as a
        screen-space disk of fixed pixel radius at its projected center,
        accumulated with alpha compositing. It is O(N * r^2) and slow; a
        real deployment would swap in a CUDA surfel rasterizer (e.g.
        DeferredGS's ``diff_surfel_rasterization``) that emits the same
        G-buffer channels.
        """
        device = camera.device
        H, W = self.image_height, self.image_width
        attrs = model.get_attributes()

        # Build the per-primitive tangent frame and perturbed shading normal.
        u, v, n0 = model.get_tangent_frame()
        shading_normal = model.get_shading_normal()

        screen_xy, depth = _project_to_screen(attrs.xyz, camera)  # (N,2), (N,1)
        # Sort back-to-front for the over operator.
        order = torch.argsort(depth.squeeze(-1), descending=True)
        screen_xy = screen_xy[order]
        depth = depth[order]
        opacity = attrs.opacity[order]
        base_color = attrs.base_color[order]
        metallic = attrs.metallic[order]
        alpha_t = attrs.alpha_t[order]
        alpha_b = attrs.alpha_b[order]
        shading_normal = shading_normal[order]
        u = u[order]
        v = v[order]
        xyz = attrs.xyz[order]

        # Initialize G-buffer.
        gb = {
            "base_color": torch.zeros(H, W, 3, device=device),
            "shading_normal": torch.zeros(H, W, 3, device=device),
            "tangent": torch.zeros(H, W, 3, device=device),
            "bitangent": torch.zeros(H, W, 3, device=device),
            "metallic": torch.zeros(H, W, 1, device=device),
            "alpha_t": torch.zeros(H, W, 1, device=device),
            "alpha_b": torch.zeros(H, W, 1, device=device),
            "view_pos": torch.zeros(H, W, 3, device=device),
            "alpha": torch.zeros(H, W, 1, device=device),
        }

        r = self.splat_radius_pixels
        r_int = int(math.ceil(r))
        # Per-primitive splat. The loop is over primitives, but the inner
        # update is vectorized over the splat's pixels. For large N this is
        # the slow part -- as noted, swap in a CUDA rasterizer here.
        for i in range(screen_xy.shape[0]):
            cx, cy = screen_xy[i, 0].item(), screen_xy[i, 1].item()
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            x0 = max(int(cx) - r_int, 0)
            x1 = min(int(cx) + r_int + 1, W)
            y0 = max(int(cy) - r_int, 0)
            y1 = min(int(cy) + r_int + 1, H)
            if x0 >= x1 or y0 >= y1:
                continue
            ys, xs = torch.meshgrid(
                torch.arange(y0, y1, device=device),
                torch.arange(x0, x1, device=device),
                indexing="ij",
            )
            dist2 = (xs.float() - cx) ** 2 + (ys.float() - cy) ** 2
            # Gaussian falloff in screen space.
            weight = torch.exp(-dist2 / (2.0 * r * r))[..., None]  # (h, w, 1)
            alpha_i = opacity[i] * weight  # (h, w, 1)
            # Over operator: accum_alpha += (1 - accum_alpha) * alpha_i
            accum_alpha_prev = gb["alpha"][y0:y1, x0:x1]
            mask = (1.0 - accum_alpha_prev) * alpha_i  # (h, w, 1)
            for name, val in [
                ("base_color", base_color[i]),
                ("shading_normal", shading_normal[i]),
                ("tangent", u[i]),
                ("bitangent", v[i]),
                ("metallic", metallic[i]),
                ("alpha_t", alpha_t[i]),
                ("alpha_b", alpha_b[i]),
                ("view_pos", xyz[i]),
                ("alpha", torch.ones(1, device=device)),
            ]:
                gb[name][y0:y1, x0:x1] = gb[name][y0:y1, x0:x1] + mask * val

        return GBuffer(
            base_color=gb["base_color"],
            shading_normal=gb["shading_normal"],
            tangent=gb["tangent"],
            bitangent=gb["bitangent"],
            metallic=gb["metallic"],
            alpha_t=gb["alpha_t"],
            alpha_b=gb["alpha_b"],
            view_pos=gb["view_pos"],
            alpha=gb["alpha"],
        )

    # --- Pass 2: per-pixel ASG shading --------------------------------------
    def _shade(self, gb: GBuffer, camera: Camera) -> torch.Tensor:
        """Evaluate the ASG BRDF per lit pixel against the envmap."""
        H, W = self.image_height, self.image_width
        device = gb.device

        # View direction: from hit position toward camera center.
        wo = _safe_normalize(camera.camera_center[None, None, :] - gb.view_pos)  # (H,W,3)
        n = _safe_normalize(gb.shading_normal)
        t = _safe_normalize(gb.tangent)
        b = _safe_normalize(gb.bitangent)

        # Only shade pixels with accumulated alpha.
        lit = (gb.alpha[..., 0] > 0.01)[..., None].float()  # (H,W,1)

        # Flatten for the BRDF loop.
        N = H * W
        wo_f = wo.reshape(N, 3)
        n_f = n.reshape(N, 3)
        t_f = t.reshape(N, 3)
        b_f = b.reshape(N, 3)
        base_f = gb.base_color.reshape(N, 3)
        metal_f = gb.metallic.reshape(N, 1)
        at_f = gb.alpha_t.reshape(N, 1)
        ab_f = gb.alpha_b.reshape(N, 1)
        lit_f = lit.reshape(N, 1)

        # Build the local hemisphere sample frame per pixel: rotate the
        # Fibonacci spiral from "z up" into (t, b, n). This is the same
        # rotation R3DG does in render_equation.cu:100-114 to align the
        # sample directions to the surface normal.
        samples = self._hemisphere_samples.to(device)  # (S, 3)
        S = samples.shape[0]
        # wi_local[s] = t*sx + b*sy + n*sz
        # (N, S, 3) = (N,1,3)*(S,1) + ...
        sx = samples[:, 0][None, :, None]  # (1, S, 1)
        sy = samples[:, 1][None, :, None]
        sz = samples[:, 2][None, :, None]
        wi = (
            t_f[:, None, :] * sx
            + b_f[:, None, :] * sy
            + n_f[:, None, :] * sz
        )  # (N, S, 3)
        wi = _safe_normalize(wi)

        # Expand BRDF inputs over the sample dimension.
        wo_e = wo_f[:, None, :].expand(N, S, 3)
        n_e = n_f[:, None, :].expand(N, S, 3)
        t_e = t_f[:, None, :].expand(N, S, 3)
        b_e = b_f[:, None, :].expand(N, S, 3)
        base_e = base_f[:, None, :].expand(N, S, 3)
        metal_e = metal_f[:, None, :].expand(N, S, 1)
        at_e = at_f[:, None, :].expand(N, S, 1)
        ab_e = ab_f[:, None, :].expand(N, S, 1)

        f_d, f_s = asg_brdf(
            wo_e, n_e, t_e, b_e, wi,
            base_e, metal_e, at_e, ab_e,
        )  # each (N, S, 3)

        # Incident radiance from the envmap, and the cosine/MC weight.
        L_i = self.envmap(wi.reshape(N * S, 3)).reshape(N, S, 3)
        cos_theta = (n_e * wi).sum(dim=-1, keepdim=True).clamp_min(0.0)
        # 2*pi / S -- hemisphere solid angle per sample (uniform).
        dw = (2.0 * math.pi / S)
        transport = L_i * cos_theta * dw  # (N, S, 3)

        rgb = (f_d + f_s) * transport  # (N, S, 3)
        rgb = rgb.sum(dim=1)  # (N, 3)

        # Mask unlit pixels to background (zero).
        rgb = rgb * lit_f
        return rgb.reshape(H, W, 3)

    def forward(
        self, model: AnisotropicGaussianModel, camera: Camera
    ) -> tuple[torch.Tensor, GBuffer]:
        """Run both passes. Returns ``(rgb, gbuffer)``."""
        gb = self._splat_reference(model, camera)
        rgb = self._shade(gb, camera)
        return rgb, gb

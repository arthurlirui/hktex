# Spline-Texture Module Interface Design

**Status**: Interface draft (signatures + docstrings, no implementation)
**Date**: 2026-09-27
**Idea**: Generalise HKTex from discrete triangular meshes to differentiable
B-spline surfaces, where each anisotropic heat kernel becomes a Gaussian on
the `(u,v)` domain with covariance `g^{-1}` (inverse first fundamental form),
closed-form and differentiable in the control points.

This document specifies the interfaces of the new modules so they can be
reviewed before implementation. It mirrors the existing HKTex layering
(`Mesh` → `HeatKernelTexture` → `HeatKernelModel` → renderer) with a parallel
spline stack, maximising reuse of the existing trainer / renderer / config
machinery.

---

## 1. Architecture overview

```
                         HKTex (existing)                  Spline-Texture (new)
                         -----------------                  --------------------
Geometric carrier  :     Mesh (trimesh)            <--->  SplineSurface (Diff-NURBS wrapper)
Appearance field   :     HeatKernelTexture(KNN)    <--->  HeatKernelTextureSpline
Model (geo+appear) :     HeatKernelModel(KNN)      <--->  SplineTextureModel
Renderer           :     RayRenderer (Mitsuba)     <--->  SplineRayRenderer (subdivide -> RayRenderer)
Density control    :     ImportancePruning /               SplineDensification
                      ErrorDensification               (operates in (u,v) domain)
```

Key substitutions:
- `Mesh` → `SplineSurface`: control points are `nn.Parameter`, partials and
  the metric tensor are closed-form via Diff-NURBS.
- Barycentric `(face_id, bary)` coordinates → parametric `(u,v)` coordinates.
- KNN-heat propagation on mesh vertices → direct metric-tensor Gaussian on
  `(u,v)` (no eigen-decomposition, no KNN graph).
- `face_ids` kwarg in `forward()` → `uv` kwarg.

---

## 2. `hktex/modules/spline_surface.py`

Wraps Diff-NURBS into the `nn.Module` carrier role that `Mesh` plays, adding
the differential-geometry quantities HKTex needs.

```python
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from hktex.utils.typing import *
from hktex.utils import BaseModule

__all__ = ["SplineSurface"]


@dataclass
class SplineSpec:
    """Tensor-product B-spline surface specification (immutable structure)."""
    degree_u: int          # p
    degree_v: int          # q
    knots_u: Tensor        # [Ku]   knot vector, fixed
    knots_v: Tensor        # [Kv]
    # control points and weights are *parameters*, not part of the spec
    n_ctrl_u: int          # number of control points along u
    n_ctrl_v: int          # number of control points along v


class SplineSurface(nn.Module):
    """Differentiable tensor-product B-spline surface.

    Plays the same carrier role as :class:`hktex.modules.mesh.Mesh` but the
    geometry is a smooth parametric surface S(u,v) with a closed-form,
    control-point-differentiable first fundamental form g(u,v).

    All geometric quantities are computed in closed form from the B-spline
    basis derivatives via the Diff-NURBS backend; nothing is discretised.
    """
    spec: SplineSpec

    # --- learnable geometry ---
    control_points: nn.Parameter    # [n_ctrl_u, n_ctrl_v, 3]
    weights: nn.Parameter           # [n_ctrl_u, n_ctrl_v]   (NURBS weights; =1 for B-spline)

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
    ) -> None: ...

    # ---- core surface evaluation ----
    def evaluate(self, uv: Float[Tensor, "P 2"]) -> Float[Tensor, "P 3"]:
        """S(u,v). Maps parameter points to 3D surface points.
        Differentiable in `control_points` and `weights`."""

    def derivatives(self, uv: Float[Tensor, "P 2"]) -> tuple[
        Float[Tensor, "P 3"],   # S
        Float[Tensor, "P 3"],   # dS/du
        Float[Tensor, "P 3"],   # dS/dv
    ]:
        """Surface point and first partial derivatives, all differentiable.
        Backed by Diff-NURBS `NURBSSurface.calc_derivs`."""

    # ---- differential geometry (the HKTex-relevant layer) ----
    def first_fundamental_form(self, uv: Float[Tensor, "P 2"]) -> Float[Tensor, "P 2 2"]:
        """g(u,v) = [[E,F],[F,G]] with
        E = dS/du . dS/du, F = dS/du . dS/dv, G = dS/dv . dS/dv.
        Closed-form; differentiable in control points. Shape [P,2,2]."""

    def metric_inverse(self, uv: Float[Tensor, "P 2"]) -> Float[Tensor, "P 2 2"]:
        """g^{-1}(u,v). Used as the base covariance of each heat kernel."""

    def normal(self, uv: Float[Tensor, "P 2"]) -> Float[Tensor, "P 3"]:
        """Unit normal n = normalize(dS/du x dS/dv)."""

    def area_element(self, uv: Float[Tensor, "P 2"]) -> Float[Tensor, "P"]:
        """sqrt(det g). Used by area regularisation and kernel normalisation."""

    # ---- rendering bridge ----
    def to_mesh(self, resolution_u: int, resolution_v: int) -> tuple[
        Float[Tensor, "V 3"],   # vertices
        Int[Tensor, "F 3"],     # faces
        Float[Tensor, "V 2"],   # per-vertex (u,v) — so a hit can recover its param
    ]:
        """Adaptively subdivide S into a dense triangular mesh for the
        existing Mitsuba ray renderer. Gradients flow back through the
        subdivision (control points -> vertex positions). `resolution_*` are
        auto-tuned by the renderer based on curvature / metric distortion."""

    # ---- optimisation toggles ----
    def freeze_geometry(self) -> None: ...
    def unfreeze_geometry(self) -> None: ...
    @property
    def geometry_optimizable(self) -> bool: ...
```

**Design notes**
- `SplineSpec` separates immutable structure (degree, knots, counts) from
  learnable parameters, mirroring how `Mesh` separates `faces` (buffer) from
  `verts` (could be parameter).
- `optimize_geometry=False` implements Stage 1 (fixed geometry, appearance
  only) — the control points become buffers.
- `to_mesh` is the only place discretisation happens, and it is for rendering
  only; the metric used by appearance is always the analytic one.

---

## 3. `hktex/modules/heat_kernel_texture_spline.py`

Parallel to `HeatKernelTextureKNN`, but kernels live in `(u,v)` and their
covariance is the inverse metric — no eigen-decomposition, no KNN graph.

```python
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from hktex.utils import BaseModule
from hktex.utils.typing import *
from .spline_surface import SplineSurface
from .heat_kernel_texture import HeatKernelTexture   # for Config inheritance

__all__ = ["HeatKernelTextureSpline"]


class HeatKernelTextureSpline(HeatKernelTexture):
    """Anisotropic heat-kernel appearance on a B-spline surface.

    Each of the G kernels is centred at a parameter-domain point
    (u_k, v_k) and its spatial extent is governed by the inverse metric
    g^{-1}(u_k, v_k) at the centre, optionally modulated by a learnable
    anisotropy matrix A_k. This replaces HKTex's discrete Laplacian
    eigen-decomposition and KNN heat propagation with a closed-form
    Gaussian on the parameter domain.
    """

    @dataclass
    class Config(HeatKernelTexture.Config):
        # kernel centres live in [0,1]^2 instead of on mesh faces
        init_uv: str = "fps"          # "fps" | "uniform" — farthest-point in (u,v) or uniform grid
        # whether to use the metric-tensor covariance (True) or a plain
        # isotropic Gaussian in parameter space (False, ablation)
        use_metric_covariance: bool = True
        # whether control points are optimised jointly (Stage 2)
        optimize_geometry: bool = False

    cfg: Config
    surface: SplineSurface

    # --- learnable kernel parameters (overrides the mesh-bound ones) ---
    _kernel_uv: nn.Parameter           # [G 2]   centres in (u,v)
    _kernel_colours: nn.Parameter      # [G D]
    _angles: nn.Parameter              # [G]     orientation of A_k
    _anisotropies: nn.Parameter        # [G]     condition number of A_k
    _sigmas: nn.Parameter              # [G]     diffusion scale
    # thresholds / sharpnesses inherited from HeatKernelTexture for the
    # soft-step kernel filter used by HKTex

    def configure(self, surface: SplineSurface, **kwargs) -> None:
        """Store the spline surface and initialise kernel centres in (u,v).
        Reuses the activation/range-enforcement machinery of
        HeatKernelTexture.configure."""

    # ---- kernel evaluation (the core replacement) ----
    def kernel_weights(
        self,
        query_uv: Float[Tensor, "P 2"],
    ) -> Float[Tensor, "P G"]:
        """Return the G kernel activations at P query parameter points.

        For each kernel k with centre c_k=(u_k,v_k), anisotropy A_k and
        scale sigma_k:

            w_k(u,v) = exp( -1/(4 sigma_k^2)  d^T  g^{-1}(c_k) A_k^{-1} g^{-1}(c_k) d )

        where d = (u,v) - c_k and g(c_k) is the first fundamental form at
        the kernel centre (computed once per optimisation step, cached).
        When use_metric_covariance is False, g^{-1} is replaced by the
        identity (isotropic-in-parameter ablation)."""

    # ---- query interface (parallels prepare_points / diffuse_heat_kernels) ----
    def query(
        self,
        query_uv: Float[Tensor, "P 2"],
    ) -> tuple[
        Float[Tensor, "P D"],            # colour
        Float[Tensor, "P G"] | None,     # per-kernel contributions (for viz/pruning)
    ]:
        """Evaluate appearance at P parameter points:
            colour(u,v) = mean_colour + sum_k w_k(u,v) * colour_k  (normalised),
        then optionally through `out_net` (inherited from HeatKernelTexture).
        This is the (u,v)-domain analogue of
        HeatKernelTextureKNN.diffuse_heat_kernels."""

    # ---- densification hooks (parallels HKTex importance pruning) ----
    def kernel_centres_uv(self) -> Float[Tensor, "G 2"]: ...
    def add_kernels(self, new_uv: Float[Tensor, "G' 2"]) -> None: ...
    def prune_kernels(self, keep_mask: Bool[Tensor, "G"]) -> None: ...
    def resample_kernel_geometry(self) -> None:
        """Recompute cached g^{-1}(c_k) for every kernel centre after the
        control points have moved (called once per optimisation step in
        Stage 2)."""

    # ---- geometry coupling ----
    @property
    def surface_area(self) -> Float[Tensor, "1"]:
        """Integral of sqrt(det g) over [0,1]^2 — used as a regulariser in
        Stage 2. Computed by quadrature on a fixed grid (gradient flows to
        control points)."""
```

**Design notes**
- Inheriting `HeatKernelTexture` reuses its colour activations, `out_net`,
  range enforcement, and save/load — only the spatial kernel machinery is
  overridden.
- `resample_kernel_geometry` is the single new bookkeeping cost of Stage 2:
  after control points move, each kernel's `g^{-1}(c_k)` is stale and must be
  recomputed. This is `O(G)` closed-form evaluations, far cheaper than the
  mesh eigen-decomposition it replaces.
- `use_metric_covariance=False` is kept as an ablation to isolate the
  contribution of the analytic metric from the contribution of the spline
  carrier itself.

---

## 4. `hktex/modules/spline_texture_model.py`

Combines `SplineSurface` + `HeatKernelTextureSpline`, paralleling
`HeatKernelModelKNN`. Registered for config-driven instantiation.

```python
import hktex
from hktex.modules.base import TextureModel
from hktex.modules.spline_surface import SplineSurface
from hktex.modules.heat_kernel_texture_spline import HeatKernelTextureSpline
from hktex.utils.typing import *

__all__ = ["SplineTextureModel"]


@hktex.register("modules.spline-texture-model")
class SplineTextureModel(TextureModel):
    """Joint differentiable B-spline geometry + heat-kernel appearance model.

    Stage 1: optimize_geometry=False — control points frozen, only kernel
             parameters and colours are learned (validates the metric-kernel
             formulation against HKTex).
    Stage 2: optimize_geometry=True  — control points and kernels optimised
             jointly from multi-view images with area/curvature regularisation
             and an appearance-geometry decoupling schedule.
    """

    @dataclass
    class Config(TextureModel.Config):
        surface: dict = field(default_factory=dict)   # SplineSurface kwargs
        texture: dict = field(default_factory=dict)   # HeatKernelTextureSpline.Config
        optimize_geometry: bool = False
        point_batching: Optional[int] = None

    cfg: Config
    surface: SplineSurface
    texture: HeatKernelTextureSpline

    def configure(self, **kwargs) -> None: ...

    # ---- TextureModel interface ----
    def requires_scene_bounds(self) -> bool: return False
    def requires_face_ids(self) -> bool: return False
        # the spline model is queried by (u,v), not by face_id

    def forward(
        self,
        uv: Float[Tensor, "P 2"],     # NOTE: uv, not pts+face_ids
        **kwargs,
    ) -> Float[Tensor, "P out_dim"]:
        """Return appearance at P parameter points. Used by the renderer's
        surface-intersection callback."""

    def post_optimizer_step(self) -> None:
        """Stage 2: clamp parameters and resample cached kernel geometry
        after the control-point update."""

    # ---- regularisation exposed to the trainer ----
    def geometry_regulariser(self) -> dict[str, Tensor]:
        """Return {'area': surface_area, 'curvature': ...} for the trainer
        to combine with the rendering loss. Empty in Stage 1."""
```

**Design notes**
- `requires_face_ids() = False` because the spline is queried by `(u,v)`,
  not by mesh face — this is the key interface difference from
  `HeatKernelModel` and lets the existing trainer accept it with a thin
  adapter.
- `geometry_regulariser` is a separate hook so the trainer stays generic;
  Stage 1 returns an empty dict and the trainer's regularisation term is a
  no-op.

---

## 5. `hktex/rendering/spline_ray_renderer.py`

Bridges the analytic spline to the existing Mitsuba `RayRenderer` via
differentiable subdivision.

```python
from hktex.rendering.ray_renderer import RayRenderer
from hktex.modules.spline_surface import SplineSurface
from hktex.utils.typing import *

__all__ = ["SplineRayRenderer"]


class SplineRayRenderer(RayRenderer):
    """Rays a B-spline surface by adaptively subdividing it to a mesh and
    delegating to the existing Mitsuba ray renderer.

    The subdivision is differentiable: vertex positions are linear
    combinations of control points via the basis matrix, so gradients of the
    image loss reach the control points through the hit vertices. Each
    vertex carries its (u,v) so a ray hit recovers the surface parameter,
    which is fed to SplineTextureModel.forward(uv=...) for appearance.
    """

    def __init__(
        self,
        surface: SplineSurface,
        texture_model,                       # SplineTextureModel
        resolution_u: int = 256,
        resolution_v: int = 256,
        adaptive: bool = True,               # refine where curvature is high
        **ray_kwargs,
    ) -> None: ...

    def render(
        self,
        viewpoint: ...,
        spp: int = 256,
    ) -> Float[Tensor, "H W C"]:
        """1. Subdivide surface -> mesh (cached, rebuilt when control points
               move beyond a tolerance).
           2. Render with RayRenderer, with a per-hit (u,v) callback that
               queries the texture model for appearance.
           3. Return the image; autograd connects it to control points and
               kernel parameters."""

    def _rebuild_mesh_if_stale(self) -> None: ...
```

---

## 6. `hktex/density_controllers/spline_densification.py`

Operates in the `(u,v)` domain — simpler than the mesh case because the
domain is a regular rectangle.

```python
from hktex.density_controllers.base import DensityController
from hktex.modules.heat_kernel_texture_spline import HeatKernelTextureSpline

__all__ = ["SplineDensification"]


class SplineDensification(DensityController):
    """Importance pruning + error-based densification of heat kernels in the
    (u,v) parameter domain.

    Pruning: drop kernels whose total activation (integral of w_k over the
    domain, approximated by quadrature) falls below a threshold.
    Densification: where the per-point rendering error is high, insert new
    kernel centres via farthest-point sampling of (u,v) restricted to the
    high-error region. Both are cleaner than the mesh versions because (u,v)
    is a flat, bounded, regular domain — no geodesic farthest-point needed.
    """

    def __init__(self, texture: HeatKernelTextureSpline, **cfg) -> None: ...
    def step(self, render_error: Float[Tensor, "P"] | None = None) -> None: ...
```

---

## 7. Config files

`configs/texture_spline.yaml` (Stage 1):
```yaml
model:
  type: modules.spline-texture-model
  optimize_geometry: false
  surface:
    # path to an initial B-spline (CAD model or fitted from a coarse mesh)
    init_from: ${data.spline_path}
    degree_u: 3
    degree_v: 3
  texture:
    n_sources: 2000
    out_dim: 3
    use_metric_covariance: true
    init_uv: fps
```

`configs/multiview_spline.yaml` (Stage 2):
```yaml
model:
  type: modules.spline-texture-model
  optimize_geometry: true
  surface: { init_from: ${data.spline_path}, degree_u: 3, degree_v: 3 }
  texture: { n_sources: 2000, out_dim: 3, use_metric_covariance: true }
renderer:
  type: modules.spline-ray-renderer
  resolution_u: 256
  resolution_v: 256
  adaptive: true
regularisation:
  area_weight: 1.0e-2
  curvature_weight: 1.0e-3
optim:
  iters: 20000
  geo_appearance_schedule: alternated   # freeze-appearance / freeze-geometry windows
```

---

## 8. Implementation order

1. `SplineSurface` — wrap Diff-NURBS, unit-test `evaluate`,
   `first_fundamental_form`, `metric_inverse`, `normal` on a known surface
   (unit sphere via rational quadratic B-spline) against analytic values.
2. `HeatKernelTextureSpline.query` with `use_metric_covariance=False` first
   (isotropic-in-parameter baseline), then `True`.
3. `SplineTextureModel` Stage 1 + `texture_spline.yaml` — reproduce HKTex
   texture fitting on a fixed spline, compare against HKTex on the
   tessellated version of the same surface.
4. `SplineRayRenderer` + Stage 2 — joint optimisation on CAD-class objects.
5. `SplineDensification`.

## 9. Open questions for review

- Should `SplineSurface` support multiple patches (trimmed NURBS) now, or
  stay single-patch and defer multi-patch to future work? **Recommendation:
  single-patch for v1; the interface above is patch-agnostic so a
  `MultiPatchSplineSurface` can subclass it.**
- NURBS weights: learn them (full NURBS) or fix at 1 (B-spline)? Weights add
  expressivity (exact conics) but complicate regularisation. **Recommendation:
  fix at 1 for v1, expose as a config flag.**
- Area regularisation quadrature: fixed grid (cheap, fixed cost) vs. adaptive
  (accurate at high curvature)? **Recommendation: fixed grid sized to the
  control mesh, refined with subdivision.**

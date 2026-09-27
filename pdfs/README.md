# Related Work — PDF Library

This folder collects the arXiv PDFs of papers related to the anisotropic
Gaussian splatting integration (tangent-space normal perturbation + ASG
microfacet BRDF + deferred G-buffer splatting).

The corresponding BibTeX entries live in `../latex/references.bib`.
Each PDF is named `<arXiv-ID>.pdf`.

## File → Paper mapping

| File | Category | Paper |
|------|----------|-------|
| 2308.04079.pdf | Core GS | 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023) |
| 2403.17888.pdf | Core GS | 2D Gaussian Splatting for Geometrically Accurate Radiance Fields (Huang et al., SIGGRAPH 2024) |
| 2311.16473.pdf | Relightable / IR | GS-IR: 3D Gaussian Splatting for Inverse Rendering (Liang et al., 2023) |
| 2311.16043.pdf | Relightable / IR | Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing (Gao et al., 2023) |
| 2410.24204.pdf | Relightable / IR | GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering (Ye et al., 2024) |
| 2402.15870.pdf | Anisotropic appearance | Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian Splatting (Yang et al., SIGGRAPH 2024) |
| 2409.05868.pdf | Anisotropic appearance | SpecGaussian with Latent Features: A High-quality Modeling of the View-dependent Appearance for 3D Gaussian Splatting (Wang et al., 2024) |
| 2502.14129.pdf | Anisotropic appearance (ASG) | GlossGau: Efficient Inverse Rendering for Glossy Surface with Anisotropic Spherical Gaussian (Du et al., 2025) — **primary baseline for contribution 2** |
| 2602.13549.pdf | Anisotropic appearance (ASG) | Nighttime Autonomous Driving Scene Reconstruction with Physically-Based Gaussian Splatting (Kim et al., 2026) |
| 2604.13333.pdf | Anisotropic appearance | SSD-GS: Scattering and Shadow Decomposition for Relightable 3D Gaussian Splatting (2026) — uses anisotropic Fresnel model |
| 2508.14563.pdf | Anisotropic appearance | GOGS: High-Fidelity Geometry and Relighting for Glossy Objects via Gaussian Surfels (2025) — captures anisotropic highlights |
| 2506.13348.pdf | Normal mapping / SV materials | TextureSplat: Per-Primitive Texture Mapping for Reflective Gaussian Splatting (2025) — **closest to contribution 1** |
| 2504.06815.pdf | Normal mapping / SV materials | SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering (Sun et al., 2024) |
| 2605.05876.pdf | Deferred / surface splatting | 3DSS: 3D Surface Splatting for Inverse Rendering (2026) |
| 2404.09412.pdf | Deferred shading | DeferredGS: Decoupled and Editable Gaussian Splatting with Deferred Shading (Wu et al., 2024) — **closest to contribution 3** |
| 2410.02619.pdf | Deferred shading | GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering (2024) |
| 2411.07478.pdf | Deferred shading | GUS-IR: Gaussian Splatting with Unified Shading for Inverse Rendering (Liang et al., 2024) |
| 2507.07733.pdf | Deferred shading | RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection (Zhou et al., 2025) |
| 2504.18468.pdf | Deferred shading | RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting (Kouros et al., 2025) |
| 2510.02069.pdf | Deferred shading | Spec-Gloss Surfels and Normal-Diffuse Priors for Relightable Glossy Objects (Kouros et al., 2025) |
| 2412.19282.pdf | Reflection | Reflective Gaussian Splatting (Yao et al., 2024) |
| 2406.18544.pdf | Reflection | GS-ROR²: Bidirectional-guided 3DGS and SDF for Reflective Object Relighting and Reconstruction (Zhu et al., 2024) |
| 2509.11275.pdf | Normal priors | ROSGS: Relightable Outdoor Scenes With Gaussian Splatting (Liao et al., 2025) |
| 2504.12799.pdf | Normal priors | TSGS: Improving Gaussian Splatting for Transparent Surface Reconstruction via Normal and De-lighting Priors (Wu et al., 2025) |
| 2507.18231.pdf | Normal priors | PS-GS: Gaussian Splatting for Multi-View Photometric Stereo (Chen et al., 2025) |
| 2601.02103.pdf | Normal priors | HeadLighter: Disentangling Illumination in Generative 3D Gaussian Heads via Lightstage Captures (Wang et al., 2026) |
| 2603.14001.pdf | Physics-grounded | PhyGaP: Physically-Grounded Gaussians with Polarization Cues (Wu et al., 2026) |
| 2408.13370.pdf | Physics-grounded | BiGS: Bidirectional Gaussian Primitives for Relightable 3D Gaussian Splatting (Liu et al., 2024) |
| 2509.22112.pdf | Generation | Large Material Gaussian Model for Relightable 3D Generation (2025) |

## Notes

- All PDFs are downloaded from `https://arxiv.org/pdf/<ID>` and verified by title.
- Foundational BRDF/normal-mapping references (Cook & Torrance 1982, Walter et al. 2007 GGX, Burley 2012 Disney BRDF, NVDIFFREC CVPR 2022) are not on arXiv; they are cited in `references.bib` from their original venues and not mirrored here.

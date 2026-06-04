# Stereo Vision Depth Pipeline — 3D Reconstruction from Two Views

## Overview
Built a complete stereo vision pipeline from scratch using only NumPy and Matplotlib — no black-box computer vision libraries. Starting from a simulated 3D object, the pipeline projects it into two virtual camera views, estimates depth from disparity, and reconstructs the original 3D geometry.

## Pipeline stages

```
Simulate 3D object → Define stereo cameras → Project to left/right views
→ Calibrate cameras (DLT) → Compute disparity → Estimate depth
→ Back-project to 3D → Evaluate reconstruction
```

## Technical implementation

### 1. 3D object simulation
- Rectangular box (cuboid): W=1.0m, H=0.5m, D=2.0m
- 8 corner vertices defined in world coordinates
- Placed 2m in front of both cameras

### 2. Stereo camera setup
- Focal length: f = 200 pixels | Principal point: cx = cy = 0
- Baseline: B = 0.2m (horizontal offset between cameras)
- Both cameras share identity rotation R = I

### 3. Camera calibration — Direct Linear Transform (DLT)
Estimated 3×4 projection matrices P_left and P_right from 3D–2D correspondences:
- Built 2N×12 system of linear equations from point pairs
- Solved via SVD (last column of Vᵀ)
- **Reprojection RMSE: ~0.000000 pixels** (near-perfect on noise-free synthetic data)

### 4. Depth recovery
$$Z = \frac{f \cdot B}{\text{disparity}}$$
- Disparity = horizontal pixel shift between left and right projections
- All 8 vertices recovered with correct depth ordering (front face closer than back)

### 5. 3D reconstruction accuracy
- Back-projected all 8 vertices from (u, v, Z) using estimated intrinsics
- **RMSE: ~1e-12 m** (effectively zero — confirms pipeline correctness)

## Recovered box dimensions

| Dimension | Ground truth | Recovered |
|-----------|-------------|-----------|
| Width | 1.000 m | ~1.000 m |
| Height | 0.500 m | ~0.500 m |
| Depth | 2.000 m | ~2.000 m |

## Tech stack
`Python` `NumPy` `Matplotlib` `SciPy` `Google Colab`

---

# Data Augmentation Options

- **Geometric Transformations**
  - *Horizontal Flips*
  - *Rotation*
    - Small angles (e.g., ±10°) to avoid unrealistic perspectives.
  - *Scaling*?
    - Simulates different camera distances.

- **Color Augmentations** (for RGB frames)
  - *Brightness*, *Contrast*, *Saturation*, *Hue Jitter*...
    - Only on RGB, not depth maps.
    - Probably only 1 or 2 of them, not all

- **Noise Injection**
  - *Gaussian Noise*?
    - Simulates sensor noise, can be applied to both RGB and depth (if realistic for your renders).

- **Occlusion/Masking**
  - *Random Erasing*/*Patch Occlusion*?
    - Simulates missing regions, but ensure masks are applied identically to both RGB and depth.

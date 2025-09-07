# Z-Defocus Pro (Aperture + Visualise) for ComfyUI

**---- This node has been completly vibe-coded by Claude in order to recreate a similar tool
     to ZDefocus in Nuke. The aim is to be able to define the level of Depth of field based on
     a normalised depth-map. THIS NODE IS BY NO MEAN PERFECT and I would really appriciate the
     support and collaboration of whoever would like to refine it and fix it, I just started my
     journey and aeger to learn from experienced developers :) ----**

Depth-based defocus node with:
- **Aperture control (f-number)** for realistic DoF behavior.
- **Visualiser overlay** (green=in focus, blue=near blur, red=far blur).
- **CoC grayscale** and **in-focus mask** outputs.
- **Depth edge hardening** (quantizes depth to plateaus, killing AA/gradients at object edges).

## Install

Copy this folder to:

ComfyUI/custom_nodes/comfyui-zdefocus-pro

Restart ComfyUI (or use ComfyUI-Manager → Reload Custom Nodes).

## Inputs

- **image**: RGB IMAGE [0..1].
- **depth**: depth map [0..1] (0=near, 1=far). Use `invert_depth` if reversed.
- **focus**: depth value in focus (sample from your depth map).
- **f_number**: e.g., 1.4–22. Lower f ⇒ shallower DoF, higher f ⇒ deeper DoF.
- **max_blur_px**: max blur radius in pixels.

### Optional controls

**Focus Region Control** (⭐ Key Parameter):
- `base_focal_width` (default 0.02): **Controls how wide the in-focus region is**
  - Range: 0.0–0.25
  - Higher values = wider focus region (more area stays sharp)
  - Lower values = narrower focus region (only small area stays sharp)
  - **This is the main parameter to adjust if your focus area is too narrow/wide**

**Camera & Blur Controls**:
- `ref_f_number` (default 2.8): baseline for aperture scaling.
- `near_scale` / `far_scale`: bias blur near/far sides of focus.
- `invert_depth`, `bg_only`, `preserve_highlights`
- `num_levels` (3–8): preblur levels for multiscale blending (quality vs speed).
- `overlay_opacity`: visualiser strength.
- **Edge hardening**:
  - `harden_edges` (on/off)
  - `quantize_levels` (2–256): fewer = harder plateaus
  - `pre_smooth_px` (0–3): tiny depth blur before quantize to stabilise bins

## Performance & GPU Optimization

**Version 2.0 includes major performance improvements:**

- **Smooth blur transitions**: Fixed visible lines between blur levels with improved interpolation
- **GPU-optimized blending**: Continuous blur level interpolation using vectorized operations instead of discrete level selection
- **Memory management**: Automatic detection and handling of large images to prevent out-of-memory errors
- **Progressive processing**: For very large images, processes blur levels on-demand instead of creating full blur stacks
- **Kernel caching**: GPU kernels are cached to avoid repeated CPU→GPU transfers
- **Half-precision support**: Automatically uses FP16 for large blur operations to save memory

**Quality improvements:**
- **No more visible lines**: Smooth continuous blending between blur levels eliminates banding artifacts
- **Better large image handling**: Prevents freezing and crashes when processing high-resolution images
- **Consistent performance**: GPU-accelerated operations provide stable performance across different image sizes

**Tips for best performance:**
- Use `num_levels` between 3-8 (higher = better quality but more memory)
- For very large images (>4K), the node automatically switches to memory-efficient mode
- GPU with 8GB+ VRAM recommended for processing 4K+ images with complex blur patterns
- Lower `max_blur_px` values process faster and use less memory

## Outputs

1. **image**: defocused result (IMAGE).
2. **visualiser**: overlay (IMAGE).
3. **coc_gray**: grayscale CoC as IMAGE (3-ch).
4. **in_focus_mask**: MASK (B,H,W) in [0,1].

## Tips

### Focus Region Control
- **Too narrow focus area?** Increase `base_focal_width` from 0.02 to 0.05-0.15
- **Too wide focus area?** Decrease `base_focal_width` from 0.02 to 0.005-0.01
- **Quick test**: Start with `f_number = 2.8`, `base_focal_width = 0.02`, `max_blur_px = 16`

### Troubleshooting
- Toggle **invert_depth** if blur looks inverted.
- If you see halos at depth edges, increase `base_focal_width` slightly or reduce `max_blur_px`.
- If AA gradients in depth cause bleeding, enable **harden_edges** and try `quantize_levels = 16–32`.

## Notes

This node uses a fast multiscale blend of pre-blurred images.
For shaped bokeh and occlusion-aware edges, consider an "accurate" mode (disc kernel + occlusion)
— can be added in a future update.

# zdefocus_pro.py
# Z-Defocus Pro for ComfyUI
#
# Advanced depth-of-field simulation with realistic camera optics modeling.
# Features:
# - Physically-based aperture (f-number) calculations
# - Multiple bokeh shapes (disc, polygons, custom blade count)
# - Edge hardening for crisp depth transitions
# - Highlight preservation to prevent blown-out bokeh
# - Comprehensive depth visualizer
# - Robust tensor shape handling (BHWC/BCHW/LBCHW support)
#
# This implementation uses a pre-computed blur stack approach for performance,
# trading memory for speed by avoiding per-pixel blur radius calculations.

import math
import functools
import torch
import torch.nn.functional as F

# ---------- shape helpers ----------
def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize input to (B, C, H, W).
    Accepts: (H,W,3), (B,H,W,3), (B,3,H,W), (L,B,3,H,W) -> merges L into B.
    """
    if x.dim() == 3 and x.size(-1) in (1, 3):
        # (H,W,C) -> (1,C,H,W)
        return x.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

    if x.dim() == 4:
        # BHWC -> BCHW
        if x.size(-1) in (1, 3):
            return x.permute(0, 3, 1, 2).contiguous()
        # Assume already BCHW
        return x.contiguous()

    if x.dim() == 5:
        # (L,B,C,H,W) (our stacks) -> (L*B, C, H, W)
        if x.size(2) in (1, 3):
            L, B, C, H, W = x.shape
            return x.reshape(L * B, C, H, W).contiguous()
        # (B,L,C,H,W) -> (B*L, C, H, W)
        if x.size(1) not in (1, 3) and x.size(2) in (1, 3):
            B, L, C, H, W = x.shape
            return x.permute(1, 0, 2, 3, 4).reshape(L * B, C, H, W).contiguous()

    # Fallback: try to interpret as BCHW
    return x.contiguous()

def _to_bhwc_safe(x: torch.Tensor) -> torch.Tensor:
    """Return a BHWC view regardless of incoming dims (handles 3D/4D/5D)."""
    bchw = _as_bchw(x)
    return bchw.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)

# ---------- misc helpers ----------
def _ensure_depth01(d: torch.Tensor, invert: bool):
    """
    Normalize depth input to (B,1,H,W) format in [0,1] range.
    Handles RGB depth maps by converting to grayscale using standard luminance weights.

    Args:
        d: Depth tensor in various formats (H,W,C), (B,H,W,C), or (B,C,H,W)
        invert: If True, inverts depth values (far becomes near)

    Returns:
        Normalized depth tensor (B,1,H,W) in [0,1] range
    """
    # Normalize to BCHW first
    if d.dim() == 3 and d.size(-1) in (1, 3):
        d = d.unsqueeze(0).permute(0, 3, 1, 2)
    elif d.dim() == 4 and d.size(-1) in (1, 3):
        d = d.permute(0, 3, 1, 2)

    if d.size(1) == 3:
        r, g, b = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        d = 0.299 * r + 0.587 * g + 0.114 * b  # (B,1,H,W)

    d = d.clamp(0.0, 1.0)
    if invert:
        d = 1.0 - d
    # tiny pre-smooth to reduce speckle before optional hardening
    d = F.avg_pool2d(d, kernel_size=3, stride=1, padding=1)
    return d  # (B,1,H,W)

def _smoothstep01(x):
    x = x.clamp(0, 1)
    return x * x * (3.0 - 2.0 * x)

def _avg_blur_separable(t: torch.Tensor, radius_px: float):
    if radius_px <= 0:
        return t
    r = int(max(1, round(radius_px)))
    k = r * 2 + 1
    x = F.avg_pool2d(t, kernel_size=(1, k), stride=1, padding=(0, r))
    x = F.avg_pool2d(x, kernel_size=(k, 1), stride=1, padding=(r, 0))
    return x

def _gauss_blur_like(t: torch.Tensor, radius_px: float):
    if radius_px <= 0:
        return t
    x = _avg_blur_separable(t, radius_px)
    x = _avg_blur_separable(x, radius_px * 0.75)
    return x

def _blur_depth(d: torch.Tensor, radius_px: float):
    return _avg_blur_separable(d, radius_px)

def _harden_depth(d: torch.Tensor, enable: bool, levels: int, pre_smooth_px: float):
    if not enable:
        return d
    d2 = _blur_depth(d, pre_smooth_px) if pre_smooth_px > 0 else d
    L = max(2, int(levels))
    idx = torch.round(d2 * (L - 1))
    dq  = idx / (L - 1)
    return dq.clamp(0, 1)

# ---------- Bokeh kernels ----------
@functools.lru_cache(maxsize=256)
def _disc_kernel(radius: int) -> torch.Tensor:
    if radius < 1:
        return torch.tensor([[[[1.0]]]], dtype=torch.float32)
    k = 2 * radius + 1
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing="ij",
    )
    mask = (xx * xx + yy * yy) <= (radius + 0.2) ** 2
    ker = mask.to(torch.float32)
    ker /= ker.sum().clamp(min=1.0)
    return ker.unsqueeze(0).unsqueeze(0)

@functools.lru_cache(maxsize=256)
def _polygon_kernel(radius: int, sides: int, rotation_deg: float) -> torch.Tensor:
    if radius < 1:
        return torch.tensor([[[[1.0]]]], dtype=torch.float32)
    sides = max(3, int(sides))
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing="ij",
    )
    x = xx / float(max(radius, 1))
    y = yy / float(max(radius, 1))

    th = math.radians(rotation_deg % 360.0)
    cos_t, sin_t = math.cos(th), math.sin(th)
    xr = x * cos_t - y * sin_t
    yr = x * sin_t + y * cos_t

    inradius = math.cos(math.pi / sides)
    inside = torch.ones_like(xr, dtype=torch.bool)
    for kidx in range(sides):
        phi = 2.0 * math.pi * (kidx / sides)
        nx, ny = math.cos(phi), math.sin(phi)
        inside &= (nx * xr + ny * yr) <= inradius + 1e-6

    ker = inside.to(torch.float32)
    ker /= ker.sum().clamp(min=1.0)
    return ker.unsqueeze(0).unsqueeze(0)

def _apply_aperture_blur(img_any: torch.Tensor, radius_px: float,
                         bokeh_shape: str, blades: int, rotation_deg: float) -> torch.Tensor:
    """
    Apply aperture-shaped blur to simulate camera bokeh effects.

    This function creates realistic camera blur by using different aperture shapes:
    - 'gauss': Smooth Gaussian blur (typical for perfect lenses)
    - 'disc': Circular aperture (most common)
    - 'hex', 'square', 'tri', 'oct': Polygonal apertures
    - 'poly': Custom polygon with specified blade count

    Args:
        img_any: Input image in BHWC/BCHW/LBCHW format
        radius_px: Blur radius in pixels
        bokeh_shape: Aperture shape type
        blades: Number of aperture blades for 'poly' shape
        rotation_deg: Rotation angle for polygonal apertures

    Returns:
        Blurred image in BCHW format
    """
    img_bchw = _as_bchw(img_any)

    if radius_px <= 0:
        return img_bchw

    if bokeh_shape == "gauss":
        return _gauss_blur_like(img_bchw, radius_px)

    r = int(max(1, round(radius_px)))
    if bokeh_shape == "disc":
        ker = _disc_kernel(r)
    elif bokeh_shape in ("hex", "square", "tri", "oct"):
        ker = _polygon_kernel(r, {"tri": 3, "square": 4, "hex": 6, "oct": 8}[bokeh_shape], rotation_deg)
    elif bokeh_shape == "poly":
        ker = _polygon_kernel(r, blades, rotation_deg)
    else:
        return _gauss_blur_like(img_bchw, radius_px)

    ker = ker.to(device=img_bchw.device, dtype=img_bchw.dtype)
    B, C, H, W = img_bchw.shape
    ker_c = ker.repeat(C, 1, 1, 1)  # (C,1,k,k)
    pad = r
    return torch.conv2d(img_bchw, ker_c, bias=None, stride=1, padding=pad, groups=C)

# ---------- Node ----------
class ZDefocusPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
                "focus": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "f_number": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 22.0, "step": 0.1}),
                "max_blur_px": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 128.0, "step": 0.5}),
            },
            "optional": {
                "ref_f_number": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 22.0, "step": 0.1}),
                "base_focal_width": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.25, "step": 0.001}),
                "near_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "far_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "bg_only": ("BOOLEAN", {"default": False}),
                "preserve_highlights": ("BOOLEAN", {"default": True}),
                "num_levels": ("INT", {"default": 5, "min": 3, "max": 8, "step": 1}),
                "overlay_opacity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Bokeh
                "bokeh_shape": (["gauss", "disc", "hex", "square", "tri", "oct", "poly"], {"default": "disc"}),
                "blades": ("INT", {"default": 6, "min": 3, "max": 12, "step": 1}),
                "rotation_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                # Edge hardening
                "harden_edges": ("BOOLEAN", {"default": True}),
                "quantize_levels": ("INT", {"default": 24, "min": 2, "max": 256, "step": 1}),
                "pre_smooth_px": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "visualiser", "coc_gray", "in_focus_mask")
    FUNCTION = "run"
    CATEGORY = "My Nodes/DOF"

    @torch.no_grad()
    def run(
        self,
        image, depth,
        focus: float, f_number: float, max_blur_px: float,
        ref_f_number: float = 2.8, base_focal_width: float = 0.02,
        near_scale: float = 1.0, far_scale: float = 1.0,
        invert_depth: bool = False, bg_only: bool = False,
        preserve_highlights: bool = True, num_levels: int = 5,
        overlay_opacity: float = 0.6,
        bokeh_shape: str = "disc", blades: int = 6, rotation_deg: float = 0.0,
        harden_edges: bool = True, quantize_levels: int = 24, pre_smooth_px: float = 0.6,
    ):
        # Input validation - early return for edge cases
        if max_blur_px <= 0:
            # No blur requested - return original image with appropriate outputs
            img_bhwc = _to_bhwc_safe(image)
            zero_mask = torch.zeros(img_bhwc.shape[:3], device=img_bhwc.device, dtype=img_bhwc.dtype)
            ones_mask = torch.ones(img_bhwc.shape[:3], device=img_bhwc.device, dtype=img_bhwc.dtype)
            return (img_bhwc, img_bhwc, img_bhwc, ones_mask)

        # Clamp parameters to safe ranges
        f_number = max(0.1, f_number)
        ref_f_number = max(0.1, ref_f_number)
        focus = max(0.0, min(1.0, focus))
        num_levels = max(3, min(32, num_levels))  # Reasonable memory limit
        # Normalize inputs with error handling
        try:
            img = _as_bchw(image).clamp(0.0, 1.0)           # (B,3,H,W)
            d   = _ensure_depth01(depth, invert=invert_depth)  # (B,1,H,W)
        except Exception as e:
            # Fallback for malformed inputs
            raise ValueError(f"Invalid input tensor format. Image shape: {image.shape}, Depth shape: {depth.shape}. Error: {e}")

        # Validate tensor dimensions match
        if img.shape[0] != d.shape[0] or img.shape[2:] != d.shape[2:]:
            raise ValueError(f"Image and depth dimensions must match. Image: {img.shape}, Depth: {d.shape}")

        d = _harden_depth(d, enable=harden_edges, levels=quantize_levels, pre_smooth_px=pre_smooth_px)

        # Aperture / CoC
        dist = (d - focus).abs()
        f_scale_wide = max(1e-6, f_number / max(ref_f_number, 1e-6))  # wider in-focus band with higher f
        f_scale_blur = max(ref_f_number, 1e-6) / max(f_number, 1e-6)  # less blur with higher f
        focal_width = base_focal_width * f_scale_wide

        t = (dist / max(focal_width, 1e-6)).clamp(0, 1)
        focus_weight = _smoothstep01(t)

        near_mask = (d < focus).float()
        far_mask  = 1.0 - near_mask
        side_scale = near_mask * near_scale + far_mask * far_scale
        if bg_only:
            side_scale = side_scale * far_mask

        coc_norm = (focus_weight * side_scale * f_scale_blur).clamp(0.0, 1.0)
        coc_px   = (coc_norm * max_blur_px).clamp(0.0, max_blur_px)

        # Pre-blurred stack (L,B,3,H,W) - creates a pyramid of progressively blurred images
        # This approach trades memory for speed by pre-computing all blur levels
        levels = max(3, int(num_levels))
        radii = torch.linspace(0.0, max_blur_px, levels, device=img.device)

        # Memory-conscious stacking - build list first, then stack to minimize peak memory
        blurred = []
        blurred.append(img)  # Level 0: no blur

        for i, r in enumerate(radii[1:], 1):
            blur_radius = float(r.item())
            blurred_level = _apply_aperture_blur(img, blur_radius,
                                               bokeh_shape=bokeh_shape, blades=blades, rotation_deg=rotation_deg)
            blurred.append(blurred_level)

        # Stack all blur levels - this is the memory-intensive step
        stack = torch.stack(blurred, dim=0)  # (L,B,3,H,W)

        # Clear the list to free intermediate memory
        del blurred

        # Blend between nearest blur levels using bilinear interpolation
        # This creates smooth depth-of-field transitions by mixing adjacent blur levels
        idx_f = (coc_px / max(max_blur_px, 1e-6)) * (levels - 1)  # Continuous index into blur stack
        idx0 = idx_f.floor().clamp(0, levels - 1)  # Lower blur level index
        idx1 = (idx0 + 1).clamp(0, levels - 1)     # Upper blur level index
        w1 = (idx_f - idx0).unsqueeze(1)           # Interpolation weight for upper level
        w0 = 1.0 - w1                              # Interpolation weight for lower level

        def gather_level(level_idx):
            """
            Efficiently gather pixels from the blur stack based on per-pixel level indices.
            This avoids expensive indexing operations by iterating through levels and
            accumulating contributions where each pixel belongs to that level.
            """
            L = stack.shape[0]
            li = level_idx.long().clamp(0, L - 1)
            out = torch.zeros_like(img)

            # Accumulate contributions from each blur level
            for l in range(L):
                mask = (li == l).float()  # Binary mask for pixels using this blur level
                if mask.any():
                    out = out + stack[l] * mask
            return out

        out0 = gather_level(idx0)
        out1 = gather_level(idx1)
        out = out0 * w0 + out1 * w1

        # Highlight preservation - prevents blown-out bright areas in bokeh
        # This technique maintains the brightness relationship between sharp and blurred regions
        if preserve_highlights:
            # Create reference blur at moderate level for brightness comparison
            ref_blur_radius = max(1.0, max_blur_px * 0.25)
            ref = _apply_aperture_blur(img, ref_blur_radius,
                                       bokeh_shape=bokeh_shape, blades=blades, rotation_deg=rotation_deg)
            base = _apply_aperture_blur(out, ref_blur_radius,
                                        bokeh_shape=bokeh_shape, blades=blades, rotation_deg=rotation_deg)

            # Calculate brightness gain to preserve highlight intensity
            eps = 1e-6
            gain = (ref + eps) / (base + eps)
            out = out * gain.clamp(0.5, 2.0)  # Limit gain to prevent artifacts

        out = out.clamp(0.0, 1.0)

        # Create depth visualizer with color-coded regions
        # Green: in-focus areas, Blue: near blur, Red: far blur
        green = (1.0 - t).clamp(0, 1)            # In-focus regions (bright green)
        near_intensity = coc_norm * near_mask    # Near blur intensity (blue channel)
        far_intensity  = coc_norm * far_mask     # Far blur intensity (red channel)

        overlay = torch.zeros_like(img)
        overlay[:, 1:2, :, :] = green           # Green channel: focus regions
        overlay[:, 2:3, :, :] += near_intensity # Blue channel: near blur
        overlay[:, 0:1, :, :] += far_intensity  # Red channel: far blur
        overlay = overlay.clamp(0, 1)

        # Blend visualizer overlay with original image
        alpha = torch.tensor(overlay_opacity, device=img.device, dtype=img.dtype)
        vis = (1 - alpha) * img + alpha * overlay
        vis = vis.clamp(0, 1)

        # Generate additional outputs for advanced users
        coc_gray_rgb = coc_norm.repeat(1, 3, 1, 1)     # Circle of Confusion as grayscale image
        in_focus_mask = (t <= 0.5).float().squeeze(1)  # Binary mask of in-focus regions

        # Convert all outputs to BHWC format as expected by ComfyUI
        return (_to_bhwc_safe(out), _to_bhwc_safe(vis), _to_bhwc_safe(coc_gray_rgb), in_focus_mask)

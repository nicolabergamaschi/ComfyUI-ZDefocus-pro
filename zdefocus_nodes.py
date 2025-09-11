# zdefocus_nodes.py
# Modular Z-Defocus Pro nodes for ComfyUI
#
# This refactored version splits the original monolithic node into focused components:
# - ZDefocusAnalyzer: Depth analysis and CoC calculation
# - ZDefocusVisualizer: Interactive depth visualization and focus point selection
# - ZDefocusPro: Core DOF processing using pre-calculated CoC maps
# - ZDefocusLegacy: Legacy all-in-one node for backward compatibility
#
# This modular approach allows users to:
# 1. Visualize depth and experiment with focus points before processing
# 2. Reuse analysis results across multiple DOF variations
# 3. Build more complex depth-based workflows

import math
import functools
import torch
import torch.nn.functional as F

# ========== SHARED UTILITIES ==========

# ---------- Shape helpers ----------
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

# ---------- Misc helpers ----------
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
# Cache kernels on GPU to avoid repeated CPU->GPU transfers
_gpu_kernel_cache = {}

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

def _get_gpu_kernel(radius: int, shape: str, sides: int, rotation_deg: float, device: torch.device, dtype: torch.dtype):
    """Get cached GPU kernel to avoid repeated transfers."""
    cache_key = (radius, shape, sides, rotation_deg, device, dtype)

    if cache_key not in _gpu_kernel_cache:
        if shape == "disc":
            ker = _disc_kernel(radius)
        else:
            ker = _polygon_kernel(radius, sides, rotation_deg)

        # Transfer to GPU and cache
        _gpu_kernel_cache[cache_key] = ker.to(device=device, dtype=dtype)

    return _gpu_kernel_cache[cache_key]

def _apply_aperture_blur(img_any: torch.Tensor, radius_px: float,
                         bokeh_shape: str, blades: int, rotation_deg: float) -> torch.Tensor:
    """
    Apply aperture-shaped blur to simulate camera bokeh effects.
    Optimized for GPU performance with memory-efficient processing.

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

    # GPU optimization: use half precision for large blur kernels to save memory
    use_half = radius_px > 32 and img_bchw.dtype == torch.float32
    if use_half:
        original_dtype = img_bchw.dtype
        img_bchw = img_bchw.half()

    if bokeh_shape == "gauss":
        result = _gauss_blur_like(img_bchw, radius_px)
    else:
        r = int(max(1, round(radius_px)))

        # Memory optimization: limit kernel size for very large radii
        max_kernel_size = 128  # Reasonable limit for GPU memory
        if r > max_kernel_size:
            # Use separable approximation for very large blur
            result = _gauss_blur_like(img_bchw, radius_px)
        else:
            if bokeh_shape == "disc":
                ker = _get_gpu_kernel(r, "disc", 6, 0.0, img_bchw.device, img_bchw.dtype)
            elif bokeh_shape in ("hex", "square", "tri", "oct"):
                sides = {"tri": 3, "square": 4, "hex": 6, "oct": 8}[bokeh_shape]
                ker = _get_gpu_kernel(r, "poly", sides, rotation_deg, img_bchw.device, img_bchw.dtype)
            elif bokeh_shape == "poly":
                ker = _get_gpu_kernel(r, "poly", blades, rotation_deg, img_bchw.device, img_bchw.dtype)
            else:
                result = _gauss_blur_like(img_bchw, radius_px)
                if use_half:
                    result = result.to(original_dtype)
                return result

            B, C, H, W = img_bchw.shape
            ker_c = ker.repeat(C, 1, 1, 1)  # (C,1,k,k)
            pad = r

            # GPU optimization: use groups convolution for efficiency
            result = torch.conv2d(img_bchw, ker_c, bias=None, stride=1, padding=pad, groups=C)

    # Convert back to original precision if needed
    if use_half:
        result = result.to(original_dtype)

    return result

# ========== MODULAR NODES ==========

class ZDefocusAnalyzer:
    """
    Analyzes depth maps and calculates Circle of Confusion (CoC) values.

    This node performs the depth-to-blur calculation without applying the actual blur,
    allowing users to experiment with focus settings and visualize the results before
    committing to the expensive blur operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
                "focus": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "f_number": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 22.0, "step": 0.1}),
                "max_blur_px": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 128.0, "step": 0.5}),
            },
            "optional": {
                # Focus Control (key parameters for focus region width)
                "base_focal_width": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.25, "step": 0.001}),
                "ref_f_number": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 22.0, "step": 0.1}),

                # Blur Scaling
                "near_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "far_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),

                # Depth Processing
                "invert_depth": ("BOOLEAN", {"default": False}),
                "bg_only": ("BOOLEAN", {"default": False}),
                "harden_edges": ("BOOLEAN", {"default": True}),
                "quantize_levels": ("INT", {"default": 24, "min": 2, "max": 256, "step": 1}),
                "pre_smooth_px": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "FLOAT")
    RETURN_NAMES = ("coc_map", "in_focus_mask", "near_mask", "effective_focus")
    FUNCTION = "analyze"
    CATEGORY = "My Nodes/DOF"

    @torch.no_grad()
    def analyze(
        self,
        depth, focus: float, f_number: float, max_blur_px: float,
        ref_f_number: float = 2.8, base_focal_width: float = 0.02,
        near_scale: float = 1.0, far_scale: float = 1.0,
        invert_depth: bool = False, bg_only: bool = False,
        harden_edges: bool = True, quantize_levels: int = 24, pre_smooth_px: float = 0.6,
    ):
        """
        Analyze depth map and calculate Circle of Confusion values.

        Returns:
            coc_map: Normalized CoC values (0=sharp, 1=max blur) as RGB image
            in_focus_mask: Binary mask of sharp regions
            near_mask: Binary mask of foreground regions
            effective_focus: The actual focus value used (for chaining)
        """
        # Parameter validation
        f_number = max(0.1, f_number)
        ref_f_number = max(0.1, ref_f_number)
        focus = max(0.0, min(1.0, focus))

        # Process depth map
        d = _ensure_depth01(depth, invert=invert_depth)  # (B,1,H,W)
        d = _harden_depth(d, enable=harden_edges, levels=quantize_levels, pre_smooth_px=pre_smooth_px)

        # Calculate aperture parameters
        dist = (d - focus).abs()
        f_scale_wide = max(1e-6, f_number / max(ref_f_number, 1e-6))
        f_scale_blur = max(ref_f_number, 1e-6) / max(f_number, 1e-6)
        focal_width = base_focal_width * f_scale_wide

        # Calculate focus weight and masks
        t = (dist / max(focal_width, 1e-6)).clamp(0, 1)
        focus_weight = _smoothstep01(t)

        near_mask_tensor = (d < focus).float()
        far_mask_tensor = 1.0 - near_mask_tensor
        side_scale = near_mask_tensor * near_scale + far_mask_tensor * far_scale

        if bg_only:
            side_scale = side_scale * far_mask_tensor

        # Calculate normalized Circle of Confusion
        coc_norm = (focus_weight * side_scale * f_scale_blur).clamp(0.0, 1.0)

        # Convert to outputs
        coc_rgb = coc_norm.repeat(1, 3, 1, 1)  # Convert to RGB for visualization
        in_focus_mask = (t <= 0.5).float().squeeze(1)  # Binary mask of sharp regions
        near_mask = near_mask_tensor.squeeze(1)  # Binary mask of foreground

        return (
            _to_bhwc_safe(coc_rgb),
            in_focus_mask,
            near_mask,
            focus
        )


class ZDefocusVisualizer:
    """
    Interactive depth visualization with focus point selection.

    This node provides a comprehensive view of the depth map with color-coded regions
    showing where blur will be applied. Users can experiment with different focus
    values and see the results immediately without processing the full image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
                "focus": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "coc_map": ("IMAGE",),  # Optional pre-calculated CoC from analyzer
                "overlay_opacity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "show_focus_point": ("BOOLEAN", {"default": True}),
                "focus_point_size": ("INT", {"default": 20, "min": 5, "max": 50, "step": 1}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "colorize_depth": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("visualization", "depth_colored", "focus_region")
    FUNCTION = "visualize"
    CATEGORY = "My Nodes/DOF"

    @torch.no_grad()
    def visualize(
        self,
        image, depth, focus: float,
        coc_map=None, overlay_opacity: float = 0.6,
        show_focus_point: bool = True, focus_point_size: int = 20,
        invert_depth: bool = False, colorize_depth: bool = True,
    ):
        """
        Create depth visualization with focus indicators.

        Returns:
            visualization: Original image with depth overlay
            depth_colored: Color-coded depth map
            focus_region: Mask showing the focused area
        """
        # Normalize inputs
        img = _as_bchw(image).clamp(0.0, 1.0)
        d = _ensure_depth01(depth, invert=invert_depth)

        # If CoC map is provided, use it; otherwise calculate basic focus regions
        if coc_map is not None:
            coc_rgb = _as_bchw(coc_map)
            coc_norm = coc_rgb[:, 0:1, :, :]  # Use red channel as CoC intensity
        else:
            # Simple focus calculation for visualization
            dist = (d - focus).abs()
            focal_width = 0.02  # Default focal width
            t = (dist / focal_width).clamp(0, 1)
            coc_norm = _smoothstep01(t)

        # Create color-coded depth visualization
        if colorize_depth:
            # Apply a color map to depth values
            depth_colored = self._apply_depth_colormap(d)
        else:
            depth_colored = d.repeat(1, 3, 1, 1)

        # Create focus region overlay
        # Green: in-focus areas, Blue: near blur, Red: far blur
        near_mask = (d < focus).float()
        far_mask = 1.0 - near_mask
        in_focus = (coc_norm <= 0.1).float()  # Sharp regions

        overlay = torch.zeros_like(img)
        overlay[:, 1:2, :, :] = in_focus           # Green: in-focus
        overlay[:, 2:3, :, :] = coc_norm * near_mask  # Blue: near blur
        overlay[:, 0:1, :, :] = coc_norm * far_mask   # Red: far blur

        # Add focus point indicator
        if show_focus_point:
            overlay = self._add_focus_indicator(overlay, d, focus, focus_point_size)

        # Blend with original image
        alpha = torch.tensor(overlay_opacity, device=img.device, dtype=img.dtype)
        visualization = (1 - alpha) * img + alpha * overlay
        visualization = visualization.clamp(0, 1)

        # Focus region mask (binary)
        focus_region = (coc_norm <= 0.1).squeeze(1)

        return (
            _to_bhwc_safe(visualization),
            _to_bhwc_safe(depth_colored),
            focus_region
        )

    def _apply_depth_colormap(self, depth):
        """Apply a rainbow colormap to depth values for better visualization."""
        # Create HSV color mapping: depth -> hue
        d_flat = depth.squeeze(1)  # (B, H, W)
        hue = d_flat * 0.8  # Map to blue-red spectrum (0.8 * 2π)

        # Convert HSV to RGB
        sat = torch.ones_like(hue)
        val = torch.ones_like(hue)

        rgb = self._hsv_to_rgb(hue, sat, val)
        return rgb.unsqueeze(0) if rgb.dim() == 3 else rgb

    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB color space."""
        h = h * 6.0  # Scale hue to [0, 6]
        i = torch.floor(h).long()
        f = h - i.float()

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        # Create RGB channels
        r = torch.zeros_like(v)
        g = torch.zeros_like(v)
        b = torch.zeros_like(v)

        # Apply HSV to RGB conversion based on hue sector
        idx = (i % 6)
        r = torch.where(idx == 0, v, torch.where(idx == 1, q, torch.where(idx == 2, p, torch.where(idx == 3, p, torch.where(idx == 4, t, v)))))
        g = torch.where(idx == 0, t, torch.where(idx == 1, v, torch.where(idx == 2, v, torch.where(idx == 3, q, torch.where(idx == 4, p, p)))))
        b = torch.where(idx == 0, p, torch.where(idx == 1, p, torch.where(idx == 2, t, torch.where(idx == 3, v, torch.where(idx == 4, v, q)))))

        return torch.stack([r, g, b], dim=1)

    def _add_focus_indicator(self, overlay, depth, focus, size):
        """Add a crosshair or circle at the focus point."""
        B, C, H, W = overlay.shape

        # Find pixel coordinates closest to focus value
        focus_mask = (depth.squeeze(1) - focus).abs()

        # For batched tensors, torch.where returns (batch_idx, y_idx, x_idx)
        indices = torch.where(focus_mask == focus_mask.min())

        if len(indices) >= 2 and len(indices[0]) > 0:
            # Use first match if multiple pixels have same depth
            # For batched input: indices = (batch_indices, y_indices, x_indices)
            if len(indices) == 3:  # Batched case
                batch_idx, focus_y, focus_x = indices
                fy, fx = int(focus_y[0]), int(focus_x[0])
            else:  # Non-batched case
                focus_y, focus_x = indices
                fy, fx = int(focus_y[0]), int(focus_x[0])

            # Draw crosshair
            half_size = size // 2
            y_start = max(0, fy - half_size)
            y_end = min(H, fy + half_size + 1)
            x_start = max(0, fx - half_size)
            x_end = min(W, fx + half_size + 1)

            # Draw horizontal line
            if y_start <= fy < y_end:
                overlay[:, :, fy, x_start:x_end] = 1.0

            # Draw vertical line
            if x_start <= fx < x_end:
                overlay[:, :, y_start:y_end, fx] = 1.0

        return overlay


class ZDefocusPro:
    """
    Core depth-of-field processing node.

    This streamlined version focuses purely on applying blur based on
    pre-calculated Circle of Confusion maps, allowing for faster iteration
    and more modular workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "coc_map": ("IMAGE",),  # Pre-calculated from ZDefocusAnalyzer
                "max_blur_px": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 128.0, "step": 0.5}),
            },
            "optional": {
                "preserve_highlights": ("BOOLEAN", {"default": True}),
                "num_levels": ("INT", {"default": 5, "min": 3, "max": 8, "step": 1}),
                "bokeh_shape": (["gauss", "disc", "hex", "square", "tri", "oct", "poly"], {"default": "disc"}),
                "blades": ("INT", {"default": 6, "min": 3, "max": 12, "step": 1}),
                "rotation_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_dof"
    CATEGORY = "My Nodes/DOF"

    @torch.no_grad()
    def apply_dof(
        self,
        image, coc_map, max_blur_px: float,
        preserve_highlights: bool = True, num_levels: int = 5,
        bokeh_shape: str = "disc", blades: int = 6, rotation_deg: float = 0.0,
    ):
        """
        Apply depth-of-field blur using pre-calculated Circle of Confusion map.

        Args:
            image: Input image to blur
            coc_map: Normalized CoC values from ZDefocusAnalyzer (0=sharp, 1=max blur)
            max_blur_px: Maximum blur radius in pixels

        Returns:
            Blurred image with realistic depth-of-field effect
        """
        if max_blur_px <= 0:
            return (_to_bhwc_safe(image),)

        # Normalize inputs
        img = _as_bchw(image).clamp(0.0, 1.0)
        coc_rgb = _as_bchw(coc_map)

        # Extract CoC intensity (use red channel or convert to grayscale)
        if coc_rgb.size(1) >= 3:
            coc_norm = (coc_rgb[:, 0:1] + coc_rgb[:, 1:2] + coc_rgb[:, 2:3]) / 3.0
        else:
            coc_norm = coc_rgb[:, 0:1]

        coc_norm = coc_norm.clamp(0.0, 1.0)
        coc_px = (coc_norm * max_blur_px).clamp(0.0, max_blur_px)

        # GPU-optimized blur stack creation with memory management
        levels = max(3, min(int(num_levels), 12))  # Limit levels to prevent memory issues

        # Memory optimization: process large images in smaller chunks if needed
        B, C, H, W = img.shape
        total_pixels = B * C * H * W
        memory_limit = 2e8  # ~800MB limit for blur stack

        if total_pixels * levels > memory_limit:
            # Use progressive processing for large images
            result = self._progressive_blur(img, coc_px, max_blur_px, levels, bokeh_shape, blades, rotation_deg)
        else:
            # Standard processing for smaller images
            result = self._standard_blur(img, coc_px, max_blur_px, levels, bokeh_shape, blades, rotation_deg)

        # Highlight preservation
        if preserve_highlights:
            ref_blur_radius = max(1.0, max_blur_px * 0.25)
            ref = _apply_aperture_blur(img, ref_blur_radius,
                                       bokeh_shape=bokeh_shape, blades=blades, rotation_deg=rotation_deg)
            base = _apply_aperture_blur(result, ref_blur_radius,
                                        bokeh_shape=bokeh_shape, blades=blades, rotation_deg=rotation_deg)

            eps = 1e-6
            gain = (ref + eps) / (base + eps)
            result = result * gain.clamp(0.5, 2.0)

        result = result.clamp(0.0, 1.0)
        return (_to_bhwc_safe(result),)

    def _standard_blur(self, img, coc_px, max_blur_px, levels, bokeh_shape, blades, rotation_deg):
        """Standard blur processing for smaller images."""
        radii = torch.linspace(0.0, max_blur_px, levels, device=img.device)

        # Build blur pyramid efficiently
        blurred = [img]  # Level 0: no blur
        for i, r in enumerate(radii[1:], 1):
            blur_radius = float(r.item())
            blurred_level = _apply_aperture_blur(
                img, blur_radius,
                bokeh_shape=bokeh_shape,
                blades=blades,
                rotation_deg=rotation_deg
            )
            blurred.append(blurred_level)

        stack = torch.stack(blurred, dim=0)  # (L,B,3,H,W)
        del blurred

        return self._blend_blur_levels(img, stack, coc_px, max_blur_px, levels)

    def _progressive_blur(self, img, coc_px, max_blur_px, levels, bokeh_shape, blades, rotation_deg):
        """Memory-efficient progressive blur processing for large images."""
        # Direct interpolation approach without storing full blur stack
        radii = torch.linspace(0.0, max_blur_px, levels, device=img.device)

        # Calculate which blur levels each pixel needs
        idx_f = (coc_px / max(max_blur_px, 1e-6)) * (levels - 1)
        idx0 = idx_f.floor().clamp(0, levels - 1)
        idx1 = (idx0 + 1).clamp(0, levels - 1)
        w1 = (idx_f - idx0)
        w0 = 1.0 - w1

        # Process blur levels on-demand
        unique_levels = torch.unique(torch.cat([idx0.flatten(), idx1.flatten()]))
        blur_cache = {}

        for level in unique_levels:
            level_int = int(level.item())
            if level_int == 0:
                blur_cache[level_int] = img
            else:
                blur_radius = float(radii[level_int].item())
                blur_cache[level_int] = _apply_aperture_blur(
                    img, blur_radius, bokeh_shape, blades, rotation_deg
                )

        # Blend results
        B, C, H, W = img.shape
        out = torch.zeros_like(img)

        # Accumulate contributions from each blur level
        for level in unique_levels:
            level_int = int(level.item())
            level_tensor = torch.tensor(level_int, device=img.device)

            # Masks for this level
            mask0 = (idx0 == level_tensor).float()
            mask1 = (idx1 == level_tensor).float()

            if mask0.any():
                contribution = blur_cache[level_int] * w0.unsqueeze(1) * mask0.unsqueeze(1)
                out = out + contribution

            if mask1.any():
                contribution = blur_cache[level_int] * w1.unsqueeze(1) * mask1.unsqueeze(1)
                out = out + contribution

        return out

    def _blend_blur_levels(self, img, stack, coc_px, max_blur_px, levels):
        """Optimized GPU-based smooth blending between blur levels."""
        # Use continuous interpolation instead of discrete level selection
        idx_f = (coc_px / max(max_blur_px, 1e-6)) * (levels - 1)
        idx0 = idx_f.floor().clamp(0, levels - 1)
        idx1 = (idx0 + 1).clamp(0, levels - 1)
        w1 = (idx_f - idx0)
        w0 = 1.0 - w1

        # Get dimensions
        B, C, H, W = img.shape
        L = stack.shape[0]

        # Ensure coc_px matches the spatial dimensions of img
        if coc_px.shape[-2:] != (H, W):
            # Resize coc_px to match image spatial dimensions
            coc_px = torch.nn.functional.interpolate(
                coc_px.unsqueeze(1) if coc_px.dim() == 3 else coc_px,
                size=(H, W), mode='nearest'
            ).squeeze(1)

            # Recalculate indices with corrected coc_px
            idx_f = (coc_px / max(max_blur_px, 1e-6)) * (levels - 1)
            idx0 = idx_f.floor().clamp(0, levels - 1)
            idx1 = (idx0 + 1).clamp(0, levels - 1)
            w1 = (idx_f - idx0)
            w0 = 1.0 - w1

        # Ensure indices have the right shape and type
        idx0 = idx0.view(B, H, W).long()
        idx1 = idx1.view(B, H, W).long()
        w0 = w0.view(B, H, W)
        w1 = w1.view(B, H, W)

        # GPU-optimized blending using gather operation
        # Reshape stack for gathering: (L, B, C, H, W) -> (B, L, C, H, W)
        stack_permuted = stack.permute(1, 0, 2, 3, 4)  # (B, L, C, H, W)

        # Expand indices for gathering across channels: (B, H, W) -> (B, 1, H, W) -> (B, C, H, W)
        idx0_expanded = idx0.unsqueeze(1).expand(B, C, H, W)
        idx1_expanded = idx1.unsqueeze(1).expand(B, C, H, W)

        # Gather blur levels for each pixel
        out0 = torch.gather(stack_permuted, 1, idx0_expanded.unsqueeze(1)).squeeze(1)  # (B, C, H, W)
        out1 = torch.gather(stack_permuted, 1, idx1_expanded.unsqueeze(1)).squeeze(1)  # (B, C, H, W)

        # Blend with weights
        w0_expanded = w0.unsqueeze(1)  # (B, 1, H, W)
        w1_expanded = w1.unsqueeze(1)  # (B, 1, H, W)
        out = out0 * w0_expanded + out1 * w1_expanded

        return out


# ========== LEGACY ALL-IN-ONE NODE ==========

class ZDefocusLegacy:
    """
    Legacy all-in-one Z-Defocus Pro node for backward compatibility.

    This node preserves the original functionality and workflow for existing users
    while the new modular nodes provide enhanced flexibility for new workflows.
    """

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
        """
        Legacy all-in-one depth of field processing.

        This method combines all functionality from the modular nodes into a single
        processing pipeline, maintaining compatibility with existing workflows.
        """
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

        # Aperture / CoC calculation
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

        # GPU-optimized blur stack creation with memory management
        # This approach balances memory usage with processing speed
        levels = max(3, min(int(num_levels), 12))  # Limit levels to prevent memory issues

        # Memory optimization: check if we need progressive processing for large images
        B, C, H, W = img.shape
        total_pixels = B * C * H * W
        memory_limit = 2e8  # ~800MB limit for blur stack

        if total_pixels * levels > memory_limit:
            # Use memory-efficient approach for large images
            out = self._progressive_blur_legacy(img, coc_px, max_blur_px, levels, bokeh_shape, blades, rotation_deg)
        else:
            # Standard processing for manageable image sizes
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

            # Optimized GPU-based smooth blending between blur levels
            # Use continuous interpolation instead of discrete level selection for smoother transitions
            idx_f = (coc_px / max(max_blur_px, 1e-6)) * (levels - 1)  # Continuous index into blur stack
            idx0 = idx_f.floor().clamp(0, levels - 1)  # Lower blur level index
            idx1 = (idx0 + 1).clamp(0, levels - 1)     # Upper blur level index
            w1 = (idx_f - idx0)                        # Interpolation weight for upper level (no unsqueeze needed)
            w0 = 1.0 - w1                              # Interpolation weight for lower level

            # Efficient GPU-native interpolation using vectorized operations
            L = stack.shape[0]

            # Reshape for efficient indexing: (L,B,C,H,W) -> (L,BCHW)
            stack_reshaped = stack.view(L, B * C * H * W)

            # Create linear indices for each pixel position
            pixel_indices = torch.arange(B * C * H * W, device=img.device, dtype=torch.long)

            # Expand indices for gathering from blur levels
            idx0_expanded = idx0.view(-1).long()  # (BCHW,)
            idx1_expanded = idx1.view(-1).long()  # (BCHW,)

            # Advanced indexing for smooth GPU-based gathering
            out0_flat = stack_reshaped[idx0_expanded, pixel_indices]  # (BCHW,)
            out1_flat = stack_reshaped[idx1_expanded, pixel_indices]  # (BCHW,)

            # Reshape back to image dimensions
            out0 = out0_flat.view(B, C, H, W)
            out1 = out1_flat.view(B, C, H, W)

            # Smooth interpolation with proper broadcasting
            w0_expanded = w0.unsqueeze(1).expand_as(out0)  # Broadcast to (B,C,H,W)
            w1_expanded = w1.unsqueeze(1).expand_as(out1)  # Broadcast to (B,C,H,W)
            out = out0 * w0_expanded + out1 * w1_expanded

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

    def _progressive_blur_legacy(self, img, coc_px, max_blur_px, levels, bokeh_shape, blades, rotation_deg):
        """Memory-efficient progressive blur processing for large images in Legacy node."""
        # Direct interpolation approach without storing full blur stack
        radii = torch.linspace(0.0, max_blur_px, levels, device=img.device)

        # Calculate which blur levels each pixel needs
        idx_f = (coc_px / max(max_blur_px, 1e-6)) * (levels - 1)
        idx0 = idx_f.floor().clamp(0, levels - 1)
        idx1 = (idx0 + 1).clamp(0, levels - 1)
        w1 = (idx_f - idx0)
        w0 = 1.0 - w1

        # Process blur levels on-demand to save memory
        unique_levels = torch.unique(torch.cat([idx0.flatten(), idx1.flatten()]))
        blur_cache = {}

        # Generate only the blur levels we actually need
        for level in unique_levels:
            level_int = int(level.item())
            if level_int == 0:
                blur_cache[level_int] = img
            else:
                blur_radius = float(radii[level_int].item())
                blur_cache[level_int] = _apply_aperture_blur(
                    img, blur_radius, bokeh_shape, blades, rotation_deg
                )

        # Blend results efficiently
        B, C, H, W = img.shape
        out = torch.zeros_like(img)

        # Accumulate contributions from each blur level
        for level in unique_levels:
            level_int = int(level.item())
            level_tensor = torch.tensor(level_int, device=img.device)

            # Masks for pixels that need this blur level
            mask0 = (idx0 == level_tensor).float()
            mask1 = (idx1 == level_tensor).float()

            if mask0.any():
                contribution = blur_cache[level_int] * w0.unsqueeze(1) * mask0.unsqueeze(1)
                out = out + contribution

            if mask1.any():
                contribution = blur_cache[level_int] * w1.unsqueeze(1) * mask1.unsqueeze(1)
                out = out + contribution

        return out

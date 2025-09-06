# ComfyUI Z-Defocus Pro

Advanced depth-of-field simulation for ComfyUI with realistic camera optics modeling and modular workflow design.

## Features

### Core Capabilities
- **Physically-based aperture calculations** with f-number support
- **Multiple bokeh shapes**: disc, polygon (hex, square, triangle, octagon), custom blade count
- **Edge hardening** for crisp depth transitions
- **Highlight preservation** to prevent blown-out bokeh
- **Robust tensor handling** (BHWC/BCHW/LBCHW support)
- **Performance optimized** with blur stack pre-computation

### Modular Workflow Design
The package provides specialized nodes for maximum flexibility:

1. **Z-Defocus Analyzer** - Converts depth maps to Circle of Confusion (CoC) maps
2. **Z-Defocus Visualizer** - Interactive depth preview with focus point selection
3. **Z-Defocus Pro** - Streamlined DOF processing using pre-calculated CoC
4. **Z-Defocus Legacy** - Original all-in-one node for backward compatibility

## Node Reference

### Z-Defocus Analyzer
Analyzes depth maps and calculates blur intensity without applying actual blur.

**Inputs:**
- `depth` - Depth map (IMAGE)
- `focus` - Focus distance (0.0-1.0)
- `f_number` - Camera aperture (1.0-22.0)
- `max_blur_px` - Maximum blur radius in pixels

**Outputs:**
- `coc_map` - Circle of Confusion intensity map
- `in_focus_mask` - Binary mask of sharp regions
- `near_mask` - Binary mask of foreground objects
- `effective_focus` - Processed focus value (for chaining)

### Z-Defocus Visualizer
Provides interactive depth visualization with color-coded focus regions.

**Inputs:**
- `image` - Source image for overlay
- `depth` - Depth map
- `focus` - Focus distance
- `coc_map` (optional) - Pre-calculated CoC from Analyzer
- `overlay_opacity` - Visualization blend strength
- `show_focus_point` - Display focus crosshair
- `colorize_depth` - Apply rainbow depth coloring

**Outputs:**
- `visualization` - Image with depth overlay (Green=focus, Blue=near, Red=far)
- `depth_colored` - Color-mapped depth visualization
- `focus_region` - Binary mask of focused areas

### Z-Defocus Pro
Streamlined DOF processing using pre-calculated CoC maps.

**Inputs:**
- `image` - Source image to blur
- `coc_map` - CoC intensity from Analyzer
- `max_blur_px` - Maximum blur radius
- `bokeh_shape` - Aperture shape (gauss, disc, hex, square, tri, oct, poly)
- `preserve_highlights` - Maintain bright region intensity

**Outputs:**
- `image` - Final depth-of-field result

### Z-Defocus Legacy
Complete all-in-one node preserving the original workflow for backward compatibility.

**Inputs:** All parameters from the original implementation
**Outputs:** `image`, `visualiser`, `coc_gray`, `in_focus_mask`

## Workflow Examples

### Basic Modular Workflow
```
Image → Z-Defocus Visualizer ← Depth Map
         ↓ (preview and adjust focus)
Image → Z-Defocus Analyzer ← Depth Map
         ↓ (coc_map output)
Image → Z-Defocus Pro → Final DOF Image
```

### Advanced Multi-Variation Workflow
```
Depth Map → Z-Defocus Analyzer (focus=0.3) → CoC Map A
         → Z-Defocus Analyzer (focus=0.7) → CoC Map B

Image → Z-Defocus Pro ← CoC Map A → Result A
      → Z-Defocus Pro ← CoC Map B → Result B
```

### Legacy Workflow
For existing users, the original workflow is preserved:
```
Image + Depth Map → Z-Defocus Legacy → DOF Image + Visualizer
```

## Camera Optics Simulation

### Aperture Control
- **f_number**: Controls depth of field width (lower = shallower DOF)
- **ref_f_number**: Reference aperture for blur scaling
- **base_focal_width**: Focus region size multiplier

### Bokeh Shapes
- **gauss**: Smooth Gaussian blur (perfect lens)
- **disc**: Circular aperture (most common)
- **hex/square/tri/oct**: Polygonal apertures
- **poly**: Custom polygon with specified blade count

### Advanced Features
- **Edge hardening**: Quantizes depth for sharper transitions
- **Highlight preservation**: Maintains bright region intensity in bokeh
- **Asymmetric scaling**: Different blur for foreground vs background

## Installation

1. Clone into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nicolabergamaschi/ComfyUI-ZDefocus-pro.git
```

2. Install dependencies:
```bash
cd ComfyUI-ZDefocus-pro
pip install -r requirements.txt
```

3. Restart ComfyUI

The nodes will appear under **My Nodes/DOF** in the node browser.

## Migration Guide

### For Existing Users
- **No changes required**: Your existing workflows will continue to work using the "Z-Defocus Legacy" node
- **Optional upgrade**: Gradually migrate to modular nodes for enhanced flexibility

### For New Users
- **Start with modular workflow**: Use Analyzer → Visualizer → Pro for maximum control
- **Quick start**: Use Legacy node if you prefer the original all-in-one approach

## Performance & Quality Tips

### Depth Map Quality
- Use high-quality depth estimation (DepthAnything, Marigold, etc.)
- Consider depth map preprocessing (smoothing, edge enhancement)
- Test different depth inversion settings for your source

### Focus Selection
- Use the Visualizer to experiment with focus points before processing
- Green overlay shows sharp regions, blue/red show blur areas
- Adjust `base_focal_width` to control focus region size

### Performance Tuning
- Lower `num_levels` for faster processing (3-5 levels usually sufficient)
- Reduce `max_blur_px` if extreme blur isn't needed
- Use simpler bokeh shapes (gauss, disc) for speed

### Artistic Control
- Try asymmetric near/far scaling for stylistic effects
- Experiment with different bokeh shapes for unique looks
- Use highlight preservation for realistic bright region handling

## Technical Architecture

### Single-File Design
All functionality is contained in `zdefocus_nodes.py` with no external dependencies within the package:
- **Shared utilities**: Common functions used across all nodes
- **Modular nodes**: Specialized components for flexible workflows
- **Legacy node**: Complete backward-compatible implementation

### Memory Optimization
- **Blur stack approach**: Pre-computes all blur levels for smooth blending
- **Efficient tensor handling**: Careful management to minimize peak memory usage
- **Format flexibility**: Automatic handling of BHWC/BCHW/LBCHW inputs

## License

This project is open source. Feel free to modify and redistribute according to your needs.

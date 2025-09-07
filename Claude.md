# Claude.md - Z-Defocus Pro Architecture & Development Guide

## Project Overview

**ComfyUI Z-Defocus Pro** is a modular depth-of-field simulation system for ComfyUI that provides physically-based camera optics modeling with a flexible, component-based architecture.

## Core Philosophy

### Design Principles
1. **Modularity**: Split complex functionality into focused, single-responsibility components
2. **Reusability**: Enable CoC calculations to be reused across multiple processing variations
3. **User Experience**: Provide interactive visualization before expensive processing
4. **Performance**: Optimize for speed while maintaining quality
5. **Backward Compatibility**: Preserve existing workflows while enabling new capabilities

### Architecture Constraints

#### Single-File Design
- **All functionality in `zdefocus_nodes.py`**: No internal dependencies between files
- **Shared utilities at top**: Common functions available to all nodes
- **Zero code duplication**: Reuse functions across nodes
- **Clean separation**: Utilities → Modular Nodes → Legacy Node

#### Memory Management
- **Efficient tensor handling**: Minimize peak memory usage during processing
- **Blur stack approach**: Pre-compute levels for smooth blending
- **Format flexibility**: Auto-handle BHWC/BCHW/LBCHW inputs
- **Cleanup**: Explicit memory cleanup after large operations

## Node Architecture

### 1. ZDefocusAnalyzer
**Purpose**: Depth analysis and Circle of Confusion (CoC) calculation

**Responsibilities**:
- Convert depth maps to normalized CoC intensity maps
- Apply camera optics calculations (f-number, focal width)
- Handle depth preprocessing (hardening, smoothing, inversion)
- Output reusable CoC maps for multiple processing passes

**Input Categories**:
- Required: `depth`, `focus`, `f_number`, `max_blur_px`
- **Focus Control**: `base_focal_width` (key parameter for focus region width)
- Camera optics: `ref_f_number`, scaling factors (`near_scale`, `far_scale`)
- Depth processing: `invert_depth`, `harden_edges`, quantization

**Outputs**:
- `coc_map`: Normalized CoC intensity (IMAGE, 0=sharp, 1=max blur)
- `in_focus_mask`: Binary sharp regions (MASK)
- `near_mask`: Binary foreground regions (MASK)
- `effective_focus`: Processed focus value (FLOAT, for chaining)

### 2. ZDefocusVisualizer
**Purpose**: Interactive depth visualization and focus point selection

**Responsibilities**:
- Provide color-coded depth and focus previews
- Enable focus point experimentation before processing
- Generate depth colormaps and focus indicators
- Support both standalone and CoC-based visualization

**Input Categories**:
- Required: `image`, `depth`, `focus`
- Optional: `coc_map` (from Analyzer for enhanced visualization)
- Visualization: `overlay_opacity`, `show_focus_point`, `colorize_depth`

**Outputs**:
- `visualization`: Image with depth overlay (IMAGE)
- `depth_colored`: Rainbow-mapped depth (IMAGE)
- `focus_region`: Binary focused areas (MASK)

**Visual Coding**:
- **Green**: In-focus regions
- **Blue**: Near blur (foreground)
- **Red**: Far blur (background)
- **White crosshair**: Focus point indicator

### 3. ZDefocusPro
**Purpose**: Streamlined DOF processing using pre-calculated CoC

**Responsibilities**:
- Apply blur based on CoC intensity maps
- Generate blur stacks with different aperture shapes
- Blend between blur levels for smooth transitions
- Preserve highlights and manage bokeh characteristics

**Input Categories**:
- Required: `image`, `coc_map`, `max_blur_px`
- Quality: `num_levels`, `preserve_highlights`
- Bokeh: `bokeh_shape`, `blades`, `rotation_deg`

**Outputs**:
- `image`: Final depth-of-field result (IMAGE)

### 4. ZDefocusLegacy
**Purpose**: Backward-compatible all-in-one processing

**Responsibilities**:
- Preserve original workflow compatibility
- Combine all functionality in single node
- Maintain exact parameter compatibility
- Provide migration path for existing users

## Technical Specifications

### Camera Optics Model

#### Focus Control Parameters
The focal region width is controlled by these key parameters:

```python
# Primary control - adjust this to widen/narrow the focus region
base_focal_width = 0.02      # Range: 0.0-0.25, default: 0.02
                            # Higher = wider focus region
                            # Lower = narrower focus region

# Secondary controls
f_number = 2.8              # Camera aperture (1.0-22.0)
ref_f_number = 2.8          # Reference aperture for scaling
```

**Focus Region Width Calculation**:
```python
f_scale_wide = f_number / ref_f_number      # Focus band scaling
focal_width = base_focal_width * f_scale_wide  # Final focus region width
```

**Usage Tips**:
- **Wider focus area**: Increase `base_focal_width` (e.g., 0.05-0.15)
- **Narrower focus area**: Decrease `base_focal_width` (e.g., 0.005-0.01)
- **Aperture effect**: Higher `f_number` also widens focus region

#### Aperture Calculation
```python
f_scale_wide = f_number / ref_f_number      # Focus band scaling
f_scale_blur = ref_f_number / f_number      # Blur intensity scaling
focal_width = base_focal_width * f_scale_wide
```

#### Circle of Confusion
```python
dist = abs(depth - focus)
t = (dist / focal_width).clamp(0, 1)
focus_weight = smoothstep(t)                # Smooth falloff
coc_norm = focus_weight * side_scale * f_scale_blur
```

### Bokeh Shapes
- **gauss**: Smooth Gaussian (perfect lens simulation)
- **disc**: Circular aperture (most natural)
- **hex/square/tri/oct**: Fixed polygons
- **poly**: Custom polygon with `blades` parameter

### Blur Stack Processing
```python
levels = range(num_levels)
radii = linspace(0, max_blur_px, levels)
stack = [apply_aperture_blur(image, r) for r in radii]
result = blend_by_coc(stack, coc_map)
```

## Development Guidelines

### Code Organization

#### File Structure
```
zdefocus_nodes.py:
├── Imports and documentation
├── Shared utilities (shape helpers, blur functions, bokeh kernels)
├── ZDefocusAnalyzer (depth → CoC)
├── ZDefocusVisualizer (preview & focus)
├── ZDefocusPro (CoC → final image)
└── ZDefocusLegacy (all-in-one compatibility)
```

#### Function Naming Conventions
- **Private utilities**: `_function_name()`
- **Shape helpers**: `_as_bchw()`, `_to_bhwc_safe()`
- **Processing**: `_ensure_depth01()`, `_harden_depth()`
- **Kernels**: `_disc_kernel()`, `_polygon_kernel()`
- **Blur**: `_apply_aperture_blur()`, `_gauss_blur_like()`

### Error Handling

#### Input Validation
- **Early returns**: Handle edge cases (max_blur_px ≤ 0)
- **Parameter clamping**: Ensure valid ranges
- **Tensor validation**: Check shape compatibility
- **Graceful fallbacks**: Default to safe operations

#### Common Issues
- **Tensor dimensions**: Handle BHWC/BCHW/LBCHW variants
- **Batch processing**: Account for batched vs single tensors
- **Device placement**: Ensure tensors on same device
- **Memory management**: Clean up large intermediate tensors

### Performance Guidelines

#### Memory Optimization
- **Blur stack size**: Limit `num_levels` to reasonable range (3-8)
- **Intermediate cleanup**: `del blurred` after stacking
- **Tensor reuse**: Avoid unnecessary copies
- **Device consistency**: Keep tensors on same device

#### Speed Optimization
- **Separable filters**: Use `_avg_blur_separable()` for efficiency
- **Cached kernels**: `@functools.lru_cache` for bokeh kernels
- **Early exits**: Return immediately for no-blur cases
- **Batch operations**: Leverage tensor batching when possible

## Workflow Patterns

### Recommended Modular Workflow
```
1. Depth + Image → ZDefocusVisualizer → Preview
2. Adjust focus based on visualization
3. Depth → ZDefocusAnalyzer → CoC Map
4. Image + CoC → ZDefocusPro → Final Result
```

### Advanced Multi-Focus Workflow
```
Depth → ZDefocusAnalyzer(focus=0.2) → CoC_A
      → ZDefocusAnalyzer(focus=0.5) → CoC_B
      → ZDefocusAnalyzer(focus=0.8) → CoC_C

Image → ZDefocusPro(CoC_A) → Result_A
      → ZDefocusPro(CoC_B) → Result_B
      → ZDefocusPro(CoC_C) → Result_C
```

### Legacy Compatibility
```
Image + Depth → ZDefocusLegacy → Result + Visualizer + CoC + Mask
```

## Quality Assurance

### Testing Requirements
- **Tensor format compatibility**: Test BHWC, BCHW, LBCHW inputs
- **Batch processing**: Verify single and multi-batch handling
- **Parameter ranges**: Test edge cases and extreme values
- **Memory usage**: Monitor peak memory consumption
- **Device compatibility**: Test CPU and GPU processing

### Performance Benchmarks
- **Speed targets**: Process 1024x1024 image in <5 seconds
- **Memory limits**: Stay under 4GB peak usage for typical workflows
- **Quality standards**: Maintain smooth blur transitions
- **Compatibility**: Support PyTorch 1.12+ and ComfyUI latest

## Migration and Compatibility

### Backward Compatibility Promise
- **ZDefocusLegacy**: Maintains exact original behavior
- **Parameter preservation**: All original inputs supported
- **Output compatibility**: Same output types and formats
- **Workflow preservation**: Existing workflows continue to work

### Migration Path
1. **Phase 1**: Users can continue with ZDefocusLegacy
2. **Phase 2**: Gradual adoption of modular components
3. **Phase 3**: Advanced users leverage multi-focus workflows
4. **Future**: Consider deprecating legacy only after wide adoption

## Extension Points

### Future Enhancements
- **Custom bokeh shapes**: User-provided kernel images
- **Focus animation**: Interpolated focus pulling sequences
- **Depth preprocessing**: Additional filters and corrections
- **Performance modes**: Quality vs speed trade-offs
- **GPU optimization**: CUDA kernels for critical paths

### API Stability
- **Core utilities**: Maintain function signatures
- **Node interfaces**: Preserve INPUT_TYPES and RETURN_TYPES
- **Parameter names**: Keep consistent naming conventions
- **Output formats**: Maintain BHWC consistency

## Documentation Standards

### Code Documentation
- **Docstrings**: All public functions and classes
- **Inline comments**: Complex algorithms and performance notes
- **Type hints**: Function parameters and return types
- **Examples**: Usage patterns in critical functions

### User Documentation
- **README.md**: Comprehensive usage guide
- **WORKFLOW_EXAMPLES.md**: Practical workflow patterns
- **Parameter explanations**: Clear descriptions of all inputs
- **Troubleshooting**: Common issues and solutions

---

## Version History & Changelog

### v2.0.0 - Modular Architecture
- Split monolithic node into specialized components
- Added interactive visualization and focus selection
- Implemented reusable CoC calculation
- Maintained full backward compatibility

### v1.0.0 - Original Implementation
- Single all-in-one ZDefocusPro node
- Complete DOF processing pipeline
- Multiple bokeh shapes and camera optics
- Edge hardening and highlight preservation

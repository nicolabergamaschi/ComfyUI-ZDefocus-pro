# Modular Z-Defocus Pro Workflow Examples

This document provides practical examples of using the new modular Z-Defocus Pro nodes.

## Basic Modular Workflow

### Step 1: Visualize and Select Focus
Connect your image and depth map to **Z-Defocus Visualizer**:
```
LoadImage → Z-Defocus Visualizer ← LoadImage (depth)
                ↓
            PreviewImage
```

- Adjust the `focus` parameter while watching the visualization
- Green areas = in focus, Blue = near blur, Red = far blur
- Use `show_focus_point` to see exactly where focus is set
- Enable `colorize_depth` for rainbow depth mapping

### Step 2: Calculate Circle of Confusion
Once you've found the right focus point, connect to **Z-Defocus Analyzer**:
```
LoadImage (depth) → Z-Defocus Analyzer
                         ↓ (coc_map)
```

- Copy the `focus` value from your visualization step
- Adjust `f_number` to control depth of field width
- Fine-tune `base_focal_width` for focus region size

### Step 3: Apply Depth of Field
Finally, process the image with **Z-Defocus Pro (Modular)**:
```
LoadImage (image) → Z-Defocus Pro (Modular) ← coc_map (from Analyzer)
                         ↓
                   Final DOF Image
```

- Set `max_blur_px` to control maximum blur intensity
- Choose `bokeh_shape` for different aperture effects
- Enable `preserve_highlights` for realistic bright areas

## Advanced Multi-Focus Workflow

Create multiple DOF variations from the same depth map:

```
LoadImage (depth) → Z-Defocus Analyzer (focus=0.2) → CoC Map A
                 → Z-Defocus Analyzer (focus=0.5) → CoC Map B
                 → Z-Defocus Analyzer (focus=0.8) → CoC Map C

LoadImage (image) → Z-Defocus Pro (Modular) ← CoC Map A → Foreground Focus
                 → Z-Defocus Pro (Modular) ← CoC Map B → Middle Focus
                 → Z-Defocus Pro (Modular) ← CoC Map C → Background Focus
```

This allows you to:
- Generate focus stacks for later blending
- Compare different focus points side-by-side
- Create focus animations by interpolating between CoC maps

## Iterative Refinement Workflow

Perfect your depth of field through iteration:

```
LoadImage (depth) → Z-Defocus Visualizer ← LoadImage (image)
                         ↓ (preview)
                    Z-Defocus Analyzer
                         ↓ (coc_map)
                    Z-Defocus Pro (Modular) ← LoadImage (image)
                         ↓
                    PreviewImage (check result)
                         ↓ (if not satisfied)
                    [Adjust parameters and repeat]
```

### Iteration Tips:
1. Start with visualization to understand your depth map
2. Use analyzer to test different camera settings (f_number, focal_width)
3. Apply DOF with conservative blur settings first
4. Gradually increase blur radius and experiment with bokeh shapes
5. Fine-tune highlight preservation and edge hardening

## Creative Techniques

### Selective Focus with Masks
Combine with mask operations for artistic control:

```
Z-Defocus Analyzer → coc_map → MaskByCoC (custom node)
                                    ↓
LoadImage → DOF Processing → FeatherMask → SelectiveBlur
```

### Focus Pulling Animation
Create smooth focus transitions:

```
Depth Map → Z-Defocus Analyzer (focus=interpolated) → Animated CoC
              ↓
Image → Z-Defocus Pro (Modular) → Focus Pull Sequence
```

### Bokeh Shape Variations
Compare different aperture effects:

```
Same CoC Map → Z-Defocus Pro (bokeh=disc) → Circular Bokeh
            → Z-Defocus Pro (bokeh=hex) → Hexagonal Bokeh
            → Z-Defocus Pro (bokeh=poly, blades=8) → Octagonal Bokeh
```

## Performance Optimization

### Memory-Efficient Processing
For large images or batch processing:

1. **Lower blur levels**: Use `num_levels=3` instead of default 5
2. **Reduced blur radius**: Start with `max_blur_px=8-12` instead of 16+
3. **Simple bokeh**: Use `gauss` or `disc` instead of complex polygons
4. **Batch CoC calculation**: Reuse analyzer output for multiple images

### Quality vs Speed Balance
- **High Quality**: `num_levels=6-8`, complex bokeh shapes, highlight preservation
- **Balanced**: `num_levels=5`, disc/hex bokeh, moderate blur radius
- **Fast Preview**: `num_levels=3`, gauss blur, reduced max_blur_px

## Troubleshooting

### Common Issues and Solutions

**Blur looks inverted:**
- Check `invert_depth` setting in Z-Defocus Analyzer
- Verify depth map format (0=near vs 0=far)

**Harsh depth transitions:**
- Enable `harden_edges` in Z-Defocus Analyzer
- Increase `base_focal_width` for softer focus falloff
- Adjust `pre_smooth_px` for depth map smoothing

**Performance issues:**
- Reduce `num_levels` in Z-Defocus Pro (Modular)
- Lower `max_blur_px` value
- Use simpler bokeh shapes

**Unrealistic highlights:**
- Enable `preserve_highlights` in Z-Defocus Pro (Modular)
- Adjust image tone mapping before DOF processing
- Consider HDR preprocessing for bright scenes

## Node Chaining Tips

### Connecting Outputs
- **coc_map** from Analyzer → **coc_map** input of Pro (Modular)
- **effective_focus** can be chained to multiple Analyzers
- **in_focus_mask** and **near_mask** can drive other effects

### Parameter Synchronization
When chaining multiple nodes:
- Keep `f_number` and `max_blur_px` consistent
- Match `invert_depth` settings across all nodes
- Use the same depth preprocessing (hardening, smoothing)

### Workflow Organization
- Group related nodes for clarity
- Use Reroute nodes for clean connections
- Label connections with comments for complex workflows

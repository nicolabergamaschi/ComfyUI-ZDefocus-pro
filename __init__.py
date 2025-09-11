# __init__.py
from .zdefocus_nodes import ZDefocusAnalyzer, ZDefocusVisualizer, ZDefocusPro, ZDefocusLegacy

NODE_CLASS_MAPPINGS = {
    # New modular nodes (recommended)
    "ZDefocusAnalyzer": ZDefocusAnalyzer,
    "ZDefocusVisualizer": ZDefocusVisualizer,
    "ZDefocusPro": ZDefocusPro,

    # Legacy all-in-one node (backward compatibility)
    "ZDefocusLegacy": ZDefocusLegacy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # New modular nodes
    "ZDefocusAnalyzer": "Z-Defocus Analyzer (Depth → CoC)",
    "ZDefocusVisualizer": "Z-Defocus Visualizer (Preview & Focus)",
    "ZDefocusPro": "Z-Defocus Pro (Modular DOF)",

    # Legacy node
    "ZDefocusLegacy": "Z-Defocus Pro (Legacy All-in-One)",
}

# For ComfyUI compatibility
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

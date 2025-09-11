#!/usr/bin/env python3
"""
Simple structure test for Z-Defocus Pro nodes that doesn't require PyTorch.
This validates that the node structure is correct for ComfyUI loading.
"""

import ast
import sys

def test_node_structure():
    """Test the structure of zdefocus_nodes.py without importing it."""

    print("🔍 Testing Z-Defocus Pro node structure...")

    # Read and parse the file
    try:
        with open('zdefocus_nodes.py', 'r') as f:
            content = f.read()

        # Parse the AST
        tree = ast.parse(content)
        print("✅ File parses correctly (no syntax errors)")

    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

    # Find all class definitions
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

    # Check for required node classes
    required_classes = ['ZDefocusAnalyzer', 'ZDefocusVisualizer', 'ZDefocusPro', 'ZDefocusLegacy']
    missing_classes = []

    for required in required_classes:
        if required in classes:
            print(f"✅ Found class: {required}")
        else:
            missing_classes.append(required)
            print(f"❌ Missing class: {required}")

    # Check for required methods in each class
    print("\n🔍 Checking class methods...")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in required_classes:
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

            if 'INPUT_TYPES' in methods:
                print(f"✅ {node.name} has INPUT_TYPES method")
            else:
                print(f"❌ {node.name} missing INPUT_TYPES method")

            # Check for processing function
            processing_methods = [m for m in methods if m in ['analyze', 'visualize', 'apply_dof', 'run']]
            if processing_methods:
                print(f"✅ {node.name} has processing method: {processing_methods[0]}")
            else:
                print(f"❌ {node.name} missing processing method")

    # Test __init__.py structure
    print("\n🔍 Testing __init__.py structure...")

    try:
        with open('__init__.py', 'r') as f:
            init_content = f.read()

        if 'NODE_CLASS_MAPPINGS' in init_content:
            print("✅ __init__.py has NODE_CLASS_MAPPINGS")
        else:
            print("❌ __init__.py missing NODE_CLASS_MAPPINGS")

        if 'NODE_DISPLAY_NAME_MAPPINGS' in init_content:
            print("✅ __init__.py has NODE_DISPLAY_NAME_MAPPINGS")
        else:
            print("❌ __init__.py missing NODE_DISPLAY_NAME_MAPPINGS")

        # Check for all required imports
        for cls in required_classes:
            if cls in init_content:
                print(f"✅ {cls} referenced in __init__.py")
            else:
                print(f"❌ {cls} not referenced in __init__.py")

    except Exception as e:
        print(f"❌ Error reading __init__.py: {e}")
        return False

    if not missing_classes:
        print(f"\n🎉 SUCCESS! All {len(required_classes)} node classes found and properly structured")
        print("📝 The nodes should load correctly in ComfyUI (assuming PyTorch is available)")
        return True
    else:
        print(f"\n❌ ISSUES FOUND: Missing {len(missing_classes)} classes: {missing_classes}")
        return False

if __name__ == "__main__":
    success = test_node_structure()
    sys.exit(0 if success else 1)

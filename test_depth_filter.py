#!/usr/bin/env python3
"""
Test script to verify that ft_depth filtering is working correctly.
Checks which layers contribute to ft_depth after implementing the filter.
"""

import sys
sys.path.insert(0, 'src')

from maritime_module.core.graph import Weights
from maritime_module.utils.s57_classification import S57Classifier

# Initialize classifier and weights (without factory for this test)
classifier = S57Classifier()

# Create a minimal mock factory object
class MockFactory:
    pass

mock_factory = MockFactory()
weights = Weights(mock_factory)

# Get the feature layer configuration
config = weights.get_feature_layers_from_classifier()

# Find all layers that are configured to populate ft_depth
depth_layers = {layer: cfg for layer, cfg in config.items() if cfg.get('column') == 'ft_depth'}

print("=" * 80)
print("DEPTH LAYER FILTERING TEST")
print("=" * 80)
print()
print(f"Total layers configured to populate ft_depth: {len(depth_layers)}")
print()

if depth_layers:
    print("Layers that will populate ft_depth:")
    for layer, cfg in sorted(depth_layers.items()):
        attrs = cfg.get('attributes', [])
        agg = cfg.get('aggregation', 'N/A')

        # Get the layer classification info
        layer_upper = layer.upper()
        layer_info = classifier._classification_db.get(layer_upper, None)
        if layer_info:
            nav_class = layer_info[0].name if hasattr(layer_info[0], 'name') else str(layer_info[0])
            category = layer_info[1] if len(layer_info) > 1 else 'N/A'
            desc = layer_info[4] if len(layer_info) > 4 else 'N/A'
        else:
            nav_class = 'UNKNOWN'
            category = 'N/A'
            desc = 'N/A'

        print(f"  {layer.upper():15s} class={nav_class:15s} category={category:15s}")
        print(f"    → {desc}")
        print(f"    → attributes: {attrs}, aggregation: {agg}")
        print()
else:
    print("WARNING: No layers configured for ft_depth!")
    print()

# Expected configuration
print("=" * 80)
print("EXPECTED CONFIGURATION")
print("=" * 80)
print()
print("Layers that SHOULD populate ft_depth (navigational depths):")
print("  DEPARE  - Depth area (primary charted depths)")
print("  DRGARE  - Dredged area (maintained depths)")
print("  SWPARE  - Swept area (verified clear depths)")
print()
print("Layers that should be EXCLUDED from ft_depth (infrastructure):")
print("  BERTHS  - Berth (mooring depths, not transit)")
print("  FAIRWY  - Fairway (route depths, should use separate column)")
print("  GATCON  - Gate (may have drval1=0)")
print("  DRYDOC  - Dry dock (has drval1=0)")
print("  FLODOC  - Floating dock (infrastructure)")
print("  DWRTCL  - Deep water route centerline (route-specific)")
print("  DWRTPT  - Deep water route part (route-specific)")
print("  RCRTCL  - Recommended route centerline (route-specific)")
print("  RECTRC  - Recommended track (route-specific)")
print("  TWRTPT  - Two-way route part (route-specific)")
print("  CBLSUB  - Cable submarine (infrastructure)")
print("  PIPSOL  - Pipeline (infrastructure)")
print()

# Verify the filter is working
print("=" * 80)
print("FILTER VERIFICATION")
print("=" * 80)
print()

expected_layers = {'depare', 'drgare', 'swpare'}
actual_layers = set(depth_layers.keys())

if actual_layers == expected_layers:
    print("✓ SUCCESS: Filter is working correctly!")
    print(f"  Configured layers match expected: {sorted(actual_layers)}")
else:
    print("✗ FAILURE: Filter configuration mismatch!")
    print(f"  Expected: {sorted(expected_layers)}")
    print(f"  Actual:   {sorted(actual_layers)}")

    missing = expected_layers - actual_layers
    extra = actual_layers - expected_layers

    if missing:
        print(f"  Missing layers: {sorted(missing)}")
    if extra:
        print(f"  Extra layers (should be excluded): {sorted(extra)}")

print()

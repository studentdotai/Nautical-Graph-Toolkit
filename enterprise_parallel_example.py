#!/usr/bin/env python3
"""
Enterprise-Safe Parallel Processing Example for S57Advanced

This example demonstrates the conservative parallel processing approach
designed for enterprise applications that prioritize data integrity.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.maritime_module.core.s57_data import S57Advanced, S57AdvancedConfig

def demonstrate_enterprise_safe_parallel():
    """Demonstrate enterprise-safe parallel processing configurations."""
    
    print("=== Enterprise-Safe Parallel Processing for S57Advanced ===\n")
    
    # 1. Maximum Safety Configuration (Default)
    print("1. MAXIMUM SAFETY (Default - Sequential Processing)")
    max_safety_config = S57AdvancedConfig(
        enable_parallel_processing=False,  # Disabled by default
        enable_debug_logging=True
    )
    print(max_safety_config.get_configuration_summary())
    print()
    
    # 2. High Safety Configuration (Read-Only Parallel)
    print("2. HIGH SAFETY (Read-Only Parallel Processing)")
    high_safety_config = S57AdvancedConfig(
        enable_parallel_processing=True,
        parallel_read_only=True,           # Only parallel reads
        parallel_db_writes=False,          # No parallel writes
        parallel_validation_level='strict', # Strict error handling
        max_parallel_workers=2,            # Conservative worker count
        enable_debug_logging=True
    )
    print(high_safety_config.get_configuration_summary())
    print()
    
    # 3. Moderate Safety Configuration (Parallel with Strict Validation)
    print("3. MODERATE SAFETY (Parallel Processing with Strict Validation)")
    moderate_safety_config = S57AdvancedConfig(
        enable_parallel_processing=True,
        parallel_read_only=True,
        parallel_db_writes=True,           # Enable parallel writes
        parallel_validation_level='strict', # Maintain strict validation
        max_parallel_workers=3,
        enable_debug_logging=True
    )
    print(moderate_safety_config.get_configuration_summary())
    print()
    
    # 4. Enterprise Recommendations
    print("=== Enterprise Usage Recommendations ===")
    print()
    print("🛡️  For CRITICAL PRODUCTION SYSTEMS:")
    print("   - Use MAXIMUM SAFETY (default sequential processing)")
    print("   - Enable comprehensive logging for audit trails")
    print("   - Run parallel processing only in development/testing")
    print()
    print("🚀 For PERFORMANCE-OPTIMIZED SYSTEMS:")
    print("   - Use HIGH SAFETY (read-only parallel processing)")
    print("   - Limit to 2-4 parallel workers for stability")
    print("   - Maintain strict validation levels")
    print()
    print("⚖️  For BALANCED ENTERPRISE SYSTEMS:")
    print("   - Use MODERATE SAFETY with careful testing")
    print("   - Enable parallel processing during development")
    print("   - Fallback to sequential for critical batch jobs")
    print()
    
    # 5. Configuration Validation Example
    print("=== Configuration Validation Example ===")
    print()
    print("Testing configuration with problematic settings...")
    problematic_config = S57AdvancedConfig(
        enable_parallel_processing=True,
        parallel_read_only=False,          # Risky setting
        parallel_db_writes=True,           # Combined with above = high risk
        parallel_validation_level='minimal', # Low safety
        max_parallel_workers=12,           # Too many workers
        enable_debug_logging=True          # Show all warnings
    )
    print("Configuration created with warnings. Check logs above for enterprise safety recommendations.")
    print()
    
    # 6. Memory and Performance Estimates
    print("=== Performance Impact Analysis ===")
    print()
    configs = [
        ("Sequential (Maximum Safety)", max_safety_config),
        ("Read-Only Parallel (High Safety)", high_safety_config), 
        ("Parallel with Validation (Moderate Safety)", moderate_safety_config)
    ]
    
    for name, config in configs:
        memory_info = config.get_memory_info()
        print(f"{name}:")
        print(f"   Workers: {memory_info.get('parallel_workers', 1)}")
        print(f"   Safety Level: {memory_info.get('parallel_safety_level', 'MAXIMUM')}")
        print(f"   Expected Performance Gain: {estimate_performance_gain(config)}")
        print()

def estimate_performance_gain(config: S57AdvancedConfig) -> str:
    """Estimate performance improvement for given configuration."""
    if not config.enable_parallel_processing:
        return "Baseline (1x)"
    
    workers = config.max_parallel_workers
    if config.parallel_read_only and not config.parallel_db_writes:
        # Read-only parallel processing - modest gains
        gain = min(1.3, 1 + (workers - 1) * 0.1)  # Up to 30% improvement
        return f"~{gain:.1f}x (Conservative estimate for read-only parallel)"
    elif config.parallel_db_writes and config.parallel_validation_level == 'strict':
        # Parallel writes with strict validation - better gains but higher risk
        gain = min(1.8, 1 + (workers - 1) * 0.2)  # Up to 80% improvement
        return f"~{gain:.1f}x (Higher performance, verify database can handle concurrent writes)"
    else:
        return "Variable (depends on validation level and database capacity)"

if __name__ == "__main__":
    demonstrate_enterprise_safe_parallel()
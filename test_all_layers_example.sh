#!/bin/bash
# Example script showing how to test all available layers
# WARNING: This will take a very long time to complete!

echo "Testing ALL available layers in the ENC dataset..."
echo "This will discover and test every layer with data."
echo "Expected to take 10+ minutes depending on dataset size."
echo ""

# Set environment variable to test all layers
export TEST_ALL_LAYERS=true

# Run the test
.venv/bin/python -m pytest tests/core__real_data/test_enc_factory.py::TestENCDataFactory::test_unanimous_output_across_formats -v

echo ""
echo "All layers testing completed!"
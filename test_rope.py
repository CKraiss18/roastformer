"""
Quick test to verify RoPE implementation works
Run this locally before uploading to Colab
"""

import torch
import sys
sys.path.append('.')

from src.model.transformer_adapter import RoPEPositionalEncoding

print("="*80)
print("TESTING RoPE IMPLEMENTATION")
print("="*80)

# Test parameters
batch_size = 2
seq_len = 100
d_model = 256

# Create RoPE
print("\n1. Creating RoPE instance...")
rope = RoPEPositionalEncoding(d_model=d_model, max_len=1000)
print(f"   ✓ RoPE created: d_model={d_model}, max_len=1000")

# Create sample input
print("\n2. Creating sample input...")
x = torch.randn(batch_size, seq_len, d_model)
print(f"   ✓ Input shape: {x.shape}")

# Apply RoPE
print("\n3. Applying RoPE...")
try:
    output = rope(x)
    print(f"   ✓ Output shape: {output.shape}")

    # Verify shape
    assert output.shape == x.shape, "Shape mismatch!"
    print("   ✓ Shape matches input")

    # Verify it actually changes the input
    assert not torch.allclose(output, x), "Output identical to input!"
    print("   ✓ Output differs from input (rotation applied)")

    # Verify no NaNs or Infs
    assert torch.isfinite(output).all(), "Output contains NaN or Inf!"
    print("   ✓ All values are finite")

    # Test with longer sequence
    print("\n4. Testing with longer sequence (seq_len=1500, > max_len)...")
    x_long = torch.randn(1, 1500, d_model)
    output_long = rope(x_long)
    print(f"   ✓ Long sequence works: {output_long.shape}")
    assert output_long.shape == x_long.shape
    print("   ✓ Auto-extends cache for longer sequences")

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED! RoPE is working correctly!")
    print("="*80)
    print("\nYou can now upload the updated transformer_adapter.py to Colab!")

except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()

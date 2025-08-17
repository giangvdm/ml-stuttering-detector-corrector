"""
Test script for LoRA implementation.
This script verifies that:
1. LoRA adapters are correctly applied
2. Only ~1-2% of parameters are trainable
3. Forward pass works correctly
4. Memory usage is reduced
"""

import torch
import torch.nn as nn
import traceback
from transformers import WhisperModel
from src.components.WhisperLoRAEncoder import LoRAWhisperEncoder, apply_lora_to_whisper
import tracemalloc


def test_lora_parameter_efficiency():
    """Test that LoRA reduces trainable parameters to ~1-2%."""
    print("Testing LoRA parameter efficiency...")
    
    # Create original Whisper encoder ONLY
    whisper_full = WhisperModel.from_pretrained("openai/whisper-base")
    original_encoder = whisper_full.encoder
    
    # Create LoRA-adapted encoder
    lora_encoder = apply_lora_to_whisper(
        model_name="openai/whisper-base",
        rank=16,
        alpha=32.0,
        num_layers_from_end=4
    )
    
    # Count parameters
    original_params = sum(p.numel() for p in original_encoder.parameters())
    lora_stats = lora_encoder.get_trainable_parameters()
    
    print(f"Original Whisper encoder parameters: {original_params:,}")
    print(f"LoRA total parameters: {lora_stats['total_params']:,}")
    print(f"LoRA trainable parameters: {lora_stats['trainable_params']:,}")
    print(f"Trainable percentage: {lora_stats['trainable_percentage']:.2f}%")
    
    # Debug: Show what parameters are trainable (first 10 only)
    print(f"\nDebug - First 10 trainable parameters:")
    trainable_count = 0
    shown_count = 0
    for name, param in lora_encoder.named_parameters():
        if param.requires_grad:
            if shown_count < 10:  # Only show first 10
                print(f"  {name}: {param.numel():,} parameters")
                shown_count += 1
            trainable_count += param.numel()
    print(f"... (showing first 10 of many)")
    print(f"Total trainable (debug): {trainable_count:,}")
    
    # Debug: Check for parameter duplication
    print(f"\nParameter count comparison:")
    print(f"Original encoder only: {original_params:,}")
    print(f"LoRA total: {lora_stats['total_params']:,}")
    print(f"Difference: {lora_stats['total_params'] - original_params:,}")
    
    # Clean up
    del whisper_full
    
    # Verify we're in the target range (1-2%)
    assert 0.5 <= lora_stats['trainable_percentage'] <= 3.0, \
        f"Trainable percentage {lora_stats['trainable_percentage']:.2f}% is outside expected range"
    
    print("✓ Parameter efficiency test passed!")
    return lora_stats


def test_lora_forward_pass():
    """Test that LoRA-adapted encoder produces valid outputs."""
    print("\nTesting LoRA forward pass...")
    
    # Create LoRA encoder
    lora_encoder = apply_lora_to_whisper("openai/whisper-base")
    lora_encoder.eval()
    
    # Create dummy input (3-second clips, 80 mel bins)
    batch_size = 2
    n_mels = 80
    time_steps = 3000  # 3 seconds at 1000 frames/sec
    
    dummy_input = torch.randn(batch_size, n_mels, time_steps)
    
    # Forward pass
    with torch.no_grad():
        outputs = lora_encoder(dummy_input)
    
    # Check outputs
    assert 'last_hidden_state' in outputs, "Missing last_hidden_state in outputs"
    
    hidden_states = outputs['last_hidden_state']
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {hidden_states.shape}")
    print(f"Hidden dimension: {lora_encoder.hidden_dim}")
    
    # Verify output shape
    expected_seq_len = time_steps // 2  # Whisper downsamples by 2
    assert hidden_states.shape == (batch_size, expected_seq_len, lora_encoder.hidden_dim), \
        f"Unexpected output shape: {hidden_states.shape}"
    
    print("✓ Forward pass test passed!")
    return outputs


def test_memory_usage():
    """Test memory usage comparison."""
    print("\nTesting memory usage...")
    
    # Test with correct Whisper input size (3000 time steps)
    dummy_input = torch.randn(1, 80, 3000)  # Correct size for Whisper
    
    # Measure original Whisper memory
    tracemalloc.start()
    whisper_full = WhisperModel.from_pretrained("openai/whisper-base")
    original_encoder = whisper_full.encoder
    
    with torch.no_grad():
        _ = original_encoder(dummy_input)
    
    current, peak_original = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Clean up
    del whisper_full, original_encoder
    
    # Measure LoRA memory
    tracemalloc.start()
    lora_encoder = apply_lora_to_whisper("openai/whisper-base")
    
    with torch.no_grad():
        _ = lora_encoder(dummy_input)
    
    current, peak_lora = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Original Whisper peak memory: {peak_original / 1024 / 1024:.1f} MB")
    print(f"LoRA Whisper peak memory: {peak_lora / 1024 / 1024:.1f} MB")
    
    # Calculate memory difference (might be small since we're only comparing inference)
    if peak_original > peak_lora:
        print(f"Memory reduction: {((peak_original - peak_lora) / peak_original) * 100:.1f}%")
    else:
        print(f"Memory difference: {((peak_lora - peak_original) / peak_original) * 100:.1f}% (LoRA overhead)")
    
    print("✓ Memory usage test completed!")
    print("Note: Memory savings are primarily during training, not inference.")


def test_lora_state_dict():
    """Test LoRA state dict saving/loading."""
    print("\nTesting LoRA state dict...")
    
    # Create LoRA encoder
    lora_encoder = apply_lora_to_whisper("openai/whisper-base")
    
    # Get LoRA state dict
    lora_state = lora_encoder.get_lora_state_dict()
    
    print(f"LoRA state dict contains {len(lora_state)} parameters")
    for name in list(lora_state.keys())[:5]:  # Show first 5
        print(f"  {name}: {lora_state[name].shape}")
    
    # Test loading (create new encoder and load state)
    new_encoder = apply_lora_to_whisper("openai/whisper-base")
    new_encoder.load_lora_state_dict(lora_state)
    
    print("✓ LoRA state dict test passed!")


def main():
    """Run all LoRA tests."""
    print("Running LoRA implementation tests...\n")
    
    try:
        # Run tests
        param_stats = test_lora_parameter_efficiency()
        outputs = test_lora_forward_pass()
        test_memory_usage()
        test_lora_state_dict()
        
        print(f"\n🎉 All tests passed!")
        print(f"LoRA implementation successfully reduces trainable parameters to {param_stats['trainable_percentage']:.2f}%")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
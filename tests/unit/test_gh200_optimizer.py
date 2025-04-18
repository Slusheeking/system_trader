#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for GH200 Optimizer
------------------------------
Tests the functionality of the GH200 optimizer with a simple model.
"""

import os
import numpy as np
import tensorflow as tf
from models.optimization.gh200_optimizer import GH200Optimizer, optimize_keras_for_gh200

def create_simple_model():
    """Create a simple Keras model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main test function."""
    print("Creating test model...")
    model = create_simple_model()
    
    print("Model summary:")
    model.summary()
    
    # Create output directory if it doesn't exist
    os.makedirs('models/optimized', exist_ok=True)
    
    # Initialize the optimizer
    print("\nInitializing GH200Optimizer...")
    optimizer = GH200Optimizer(
        memory_fraction=0.5,
        enable_mixed_precision=True
    )
    
    # Print GPU info
    print(f"\nGPU Info: {optimizer.gpu_info}")
    
    # Test optimal batch size calculation
    input_shape = (10,)
    optimal_batch_size = optimizer.get_optimal_batch_size(
        input_shape=input_shape,
        model_type='keras',
        base_batch_size=32
    )
    print(f"\nOptimal batch size for input shape {input_shape}: {optimal_batch_size}")
    
    # Test memory-optimized model
    print("\nApplying memory optimizations...")
    try:
        memory_optimized_model = optimizer._optimize_keras_memory(model)
        print("\nMemory optimizations applied successfully")
        
        # Test batch size calculation
        print("\nTesting batch size calculation...")
        for shape in [(10,), (100,), (1000,)]:
            batch_size = optimizer.get_optimal_batch_size(
                input_shape=shape,
                model_type='keras',
                base_batch_size=32
            )
            print(f"Optimal batch size for input shape {shape}: {batch_size}")
        
        # Save the model directly
        print("\nSaving model directly...")
        save_path = 'models/optimized/test_model.keras'
        memory_optimized_model.save(save_path)
        print(f"Model saved to: {save_path}")
        
        print("\nGH200 optimizer functionality verified successfully")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()

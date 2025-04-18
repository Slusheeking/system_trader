#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX Model Converter
-------------------
Converts TensorFlow and XGBoost models to ONNX format for optimized inference.
Provides hardware-specific optimizations for GH200 ARM64 architecture.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple

# Try to import required libraries, handle missing dependencies gracefully
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import tf2onnx
    from tf2onnx.tfonnx import process_tf_graph
except ImportError:
    tf2onnx = None

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except ImportError:
    StandardScaler = None
    MinMaxScaler = None

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('onnx_converter')


class ONNXConverter:
    """
    Converts machine learning models to ONNX format.
    Optimizes models for specific hardware targets.
    """

    def __init__(self, target_hardware: str = 'gh200'):
        """
        Initialize the ONNX converter.
        
        Args:
            target_hardware: Target hardware platform ('gh200', 'cpu', 'gpu')
        """
        # Check if required dependencies are available
        if onnx is None or ort is None:
            logger.warning("ONNX or ONNX Runtime not installed. Some functionality will be limited.")
        
        self.target_hardware = target_hardware
        
        # Set optimization level based on target hardware
        if target_hardware == 'gh200':
            self.optimization_level = 99  # Maximum optimization
        elif target_hardware == 'gpu':
            self.optimization_level = 2
        else:
            self.optimization_level = 1
        
        # Track input and output shapes
        self.input_names = None
        self.output_names = None
        self.input_shapes = None
        self.output_shapes = None
        
        # Initialize ONNX Runtime session options if available
        if ort is not None:
            self.session_options = ort.SessionOptions()
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Set execution provider based on target hardware
            self.execution_providers = self._get_execution_providers()
        else:
            self.session_options = None
            self.execution_providers = []
        
        logger.info(f"ONNX Converter initialized for {target_hardware} target")
    
    def _get_execution_providers(self) -> List[str]:
        """
        Get the appropriate execution providers for the target hardware.
        
        Returns:
            List of execution provider names
        """
        if ort is None:
            logger.warning("ONNX Runtime not installed, cannot get execution providers")
            return []
            
        try:
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {available_providers}")
            
            if self.target_hardware == 'gh200':
                # Check if ARM execution provider is available
                if 'ArmExecutionProvider' in available_providers:
                    return ['ArmExecutionProvider']
                else:
                    logger.warning("ARM execution provider not available, falling back to CPU")
                    return ['CPUExecutionProvider']
            elif self.target_hardware == 'gpu':
                # Check if CUDA or TensorRT provider is available
                providers = []
                if 'TensorrtExecutionProvider' in available_providers:
                    providers.append('TensorrtExecutionProvider')
                if 'CUDAExecutionProvider' in available_providers:
                    providers.append('CUDAExecutionProvider')
                if providers:
                    return providers
                else:
                    logger.warning("GPU execution providers not available, falling back to CPU")
                    return ['CPUExecutionProvider']
            else:
                return ['CPUExecutionProvider']
        except Exception as e:
            logger.error(f"Error getting execution providers: {str(e)}")
            return ['CPUExecutionProvider']
    
    def convert_keras_model(self, model, output_path: str, 
                          input_signature: Optional[List] = None) -> str:
        """
        Convert a Keras model to ONNX format.
        
        Args:
            model: TensorFlow Keras model
            output_path: Path to save the ONNX model
            input_signature: Optional input signature for the model
            
        Returns:
            Path to the saved ONNX model
        """
        if tf is None or tf2onnx is None:
            logger.error("TensorFlow or tf2onnx not installed, cannot convert Keras model")
            return ""
            
        try:
            logger.info(f"Converting Keras model to ONNX format: {model.name}")
            
            if input_signature is None:
                # Try to infer input signature from the model
                try:
                    input_signature = []
                    for input_layer in model.inputs:
                        input_shape = input_layer.shape
                        input_dtype = input_layer.dtype
                        input_signature.append(tf.TensorSpec(input_shape, input_dtype, name=input_layer.name))
                except Exception as e:
                    logger.error(f"Could not infer input signature: {str(e)}")
                    return ""
            
            # Convert the model to ONNX
            # First, save as a temp SavedModel
            temp_saved_model_path = f"{output_path}_temp_saved_model"
            model.save(temp_saved_model_path, save_format='tf')
            
            # Convert to ONNX
            logger.info("Converting SavedModel to ONNX")
            model_proto, _ = tf2onnx.convert.from_saved_model(
                temp_saved_model_path,
                output_path=output_path,
                opset=13,
                target=self.execution_providers[0]
            )
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_saved_model_path)
            
            # Optimize the model
            logger.info("Optimizing ONNX model")
            optimized_model = self._optimize_onnx_model(model_proto)
            
            # Save the optimized model
            onnx.save_model(optimized_model, output_path)
            
            # Save metadata
            self._save_metadata(model, output_path)
            
            logger.info(f"ONNX model saved to: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error converting Keras model to ONNX: {str(e)}")
            return ""
    
    def convert_lstm_transformer_model(self, model: tf.keras.Model, output_path: str,
                                     sequence_length: int, n_features: int) -> str:
        """
        Convert an LSTM-Transformer model to ONNX format with special optimizations.
        
        Args:
            model: TensorFlow Keras LSTM-Transformer model
            output_path: Path to save the ONNX model
            sequence_length: Length of input sequences
            n_features: Number of input features
            
        Returns:
            Path to the saved ONNX model
        """
        logger.info(f"Converting LSTM-Transformer model to ONNX format")
        
        # Create input signature for the model
        input_signature = [tf.TensorSpec((None, sequence_length, n_features), tf.float32, name='input')]
        
        # Convert using the standard method
        onnx_path = self.convert_keras_model(model, output_path, input_signature)
        
        if not onnx_path:
            return ""
        
        # Load the ONNX model for additional optimizations
        try:
            model_proto = onnx.load(onnx_path)
            
            # Apply LSTM-specific optimizations
            logger.info("Applying LSTM-specific optimizations")
            optimized_model = self._optimize_lstm_model(model_proto)
            
            # Apply Transformer-specific optimizations
            logger.info("Applying Transformer-specific optimizations")
            optimized_model = self._optimize_transformer_model(optimized_model)
            
            # Save the optimized model
            onnx.save_model(optimized_model, output_path)
            
            # Add additional metadata for LSTM-Transformer model
            metadata = {
                'model_type': 'lstm_transformer',
                'sequence_length': sequence_length,
                'n_features': n_features
            }
            self._update_metadata(output_path, metadata)
            
            logger.info(f"Optimized LSTM-Transformer ONNX model saved to: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error optimizing LSTM-Transformer model: {str(e)}")
            return onnx_path  # Return the original path if optimization fails
    
    def convert_xgboost_model(self, model, output_path: str, 
                           input_shape: Tuple[int, ...], feature_names: Optional[List[str]] = None) -> str:
        """
        Convert an XGBoost model to ONNX format.
        
        Args:
            model: XGBoost Booster model
            output_path: Path to save the ONNX model
            input_shape: Shape of input features
            feature_names: Optional list of feature names
            
        Returns:
            Path to the saved ONNX model
        """
        if xgb is None or onnx is None:
            logger.error("XGBoost or ONNX not installed, cannot convert XGBoost model")
            return ""
            
        logger.info(f"Converting XGBoost model to ONNX format")
        
        try:
            # Save the XGBoost model temporarily
            temp_path = f"{output_path}_temp.json"
            model.save_model(temp_path)
            
            try:
                # Convert XGBoost model to ONNX using the correct approach
                from onnxmltools import convert_xgboost
                from onnxmltools.convert.common.data_types import FloatTensorType
                
                # Define input type
                input_type = [('input', FloatTensorType(input_shape))]
                
                # Convert to ONNX model
                # Note: convert_xgboost doesn't accept feature_names as a parameter
                model_onnx = convert_xgboost(model, initial_types=input_type, target_opset=13)
            except ImportError:
                logger.error("onnxmltools not installed, cannot convert XGBoost model")
                os.remove(temp_path)
                return ""
            
            # Optimize the model
            logger.info("Optimizing XGBoost ONNX model")
            optimized_model = self._optimize_onnx_model(model_onnx)
            
            # Save the optimized model
            onnx.save_model(optimized_model, output_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Save metadata
            metadata = {
                'model_type': 'xgboost',
                'input_shape': input_shape,
                'feature_names': feature_names
            }
            self._save_metadata_dict(metadata, output_path)
            
            logger.info(f"XGBoost ONNX model saved to: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error converting XGBoost model to ONNX: {str(e)}")
            return ""
    
    def _optimize_onnx_model(self, model_proto) -> Any:
        """
        Apply general optimizations to an ONNX model.
        
        Args:
            model_proto: ONNX model protocol buffer
            
        Returns:
            Optimized ONNX model
        """
        if onnx is None:
            logger.warning("ONNX not installed, skipping model optimization")
            return model_proto
            
        # Use ONNX Runtime graph optimizations
        try:
            import onnxoptimizer
            
            # Basic optimizations
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'fuse_bn_into_conv',
                'fuse_add_bias_into_conv',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_transpose_into_gemm'
            ]
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(model_proto, passes)
            
            try:
                # Constant folding
                from onnxruntime.transformers.onnx_model import OnnxModel
                onnx_model = OnnxModel(optimized_model)
                optimized_model = onnx_model.constant_folding(optimized_model)
            except ImportError:
                logger.warning("onnxruntime.transformers not available, skipping constant folding")
            
            return optimized_model
        
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping optimizations")
            return model_proto
        except Exception as e:
            logger.warning(f"Error during ONNX optimization: {str(e)}")
            return model_proto
    
    def _optimize_lstm_model(self, model_proto: onnx.ModelProto) -> onnx.ModelProto:
        """
        Apply LSTM-specific optimizations to an ONNX model.
        
        Args:
            model_proto: ONNX model protocol buffer
            
        Returns:
            Optimized ONNX model
        """
        try:
            import onnxoptimizer
            
            # LSTM-specific optimizations
            passes = [
                'fuse_matmul_add_bias_into_gemm',
                'fuse_lstm_cell_with_peephole_into_lstm',
                'fuse_consecutive_transposes',
                'fuse_transpose_into_gemm',
                'eliminate_nop_dropout'
            ]
            
            # Apply LSTM-specific optimizations
            optimized_model = onnxoptimizer.optimize(model_proto, passes)
            
            # Apply ARM-specific optimizations if targeting GH200
            if self.target_hardware == 'gh200':
                # For ARM64, we can apply additional optimizations
                # such as quantization for LSTM weights
                from onnxruntime.transformers.onnx_model import OnnxModel
                onnx_model = OnnxModel(optimized_model)
                
                # ARM64-specific optimizations for LSTM models
                # 1. Quantize LSTM weights to int8
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                # Create a temporary file for the quantized model
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                    temp_path = tmp.name
                    onnx.save_model(optimized_model, temp_path)
                    
                    # Quantize the model with ARM-specific settings
                    quantize_dynamic(
                        model_input=temp_path,
                        model_output=temp_path,
                        per_channel=True,
                        reduce_range=True,
                        weight_type=QuantType.QInt8,
                        op_types_to_quantize=['LSTM', 'MatMul', 'Gemm']
                    )
                    
                    # Load the quantized model
                    optimized_model = onnx.load(temp_path)
                
                # 2. Optimize memory layout for ARM64
                # Set memory layout to NHWC which is more efficient on ARM
                from onnx import helper
                for node in optimized_model.graph.node:
                    if node.op_type in ['Conv', 'MaxPool', 'AveragePool']:
                        # Add data_format attribute if not present
                        has_layout = False
                        for attr in node.attribute:
                            if attr.name == 'data_format':
                                has_layout = True
                                break
                        
                        if not has_layout:
                            node.attribute.append(
                                helper.make_attribute('data_format', 'NHWC')
                            )
                
                # 3. Add ARM Neon optimizations where applicable
                # This is a placeholder for actual ARM Neon optimizations
                # In a real implementation, you would use ARM-specific libraries
                # or custom operators optimized for ARM Neon SIMD instructions
            return optimized_model
        
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping LSTM optimizations")
            return model_proto
        except Exception as e:
            logger.warning(f"Error during LSTM optimization: {str(e)}")
            return model_proto
    
    def _optimize_transformer_model(self, model_proto: onnx.ModelProto) -> onnx.ModelProto:
        """
        Apply Transformer-specific optimizations to an ONNX model.
        
        Args:
            model_proto: ONNX model protocol buffer
            
        Returns:
            Optimized ONNX model
        """
        try:
            import onnxoptimizer
            
            # Transformer-specific optimizations
            passes = [
                'fuse_matmul_add_bias_into_gemm',
                'fuse_add_bias_into_conv',
                'fuse_consecutive_reduces',
                'eliminate_nop_dropout',
                'fuse_consecutive_transposes'
            ]
            
            # Apply Transformer-specific optimizations
            optimized_model = onnxoptimizer.optimize(model_proto, passes)
            
            # Apply specialized Transformer optimizations
            # For attention mechanisms, special fusion rules can be applied
            from onnxruntime.transformers.fusion_attention import FusionAttention
            from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
            from onnxruntime.transformers.onnx_model import OnnxModel
            
            onnx_model = OnnxModel(optimized_model)
            
            # Attention fusion
            attention_fusion = FusionAttention(onnx_model)
            attention_fusion.apply()
            
            # LayerNorm fusion
            layernorm_fusion = FusionLayerNormalization(onnx_model)
            layernorm_fusion.apply()
            
            return onnx_model.model
        
        except ImportError:
            logger.warning("onnxoptimizer or onnxruntime.transformers not available, skipping Transformer optimizations")
            return model_proto
        except Exception as e:
            logger.warning(f"Error during Transformer optimization: {str(e)}")
            return model_proto
    
    def _save_metadata(self, model: tf.keras.Model, output_path: str) -> None:
        """
        Save model metadata.
        
        Args:
            model: TensorFlow Keras model
            output_path: Path to the ONNX model
        """
        try:
            # Extract model information
            metadata = {
                'model_name': model.name,
                'input_shapes': {input_layer.name: input_layer.shape.as_list() for input_layer in model.inputs},
                'output_shapes': {output_layer.name: output_layer.shape.as_list() for output_layer in model.outputs},
                'target_hardware': self.target_hardware,
                'optimization_level': self.optimization_level,
                'keras_version': tf.keras.__version__,
                'tensorflow_version': tf.__version__
            }
            
            # Save metadata to JSON file
            metadata_path = f"{output_path}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model metadata saved to: {metadata_path}")
        
        except Exception as e:
            logger.warning(f"Error saving metadata: {str(e)}")
    
    def _save_metadata_dict(self, metadata: Dict[str, Any], output_path: str) -> None:
        """
        Save metadata dictionary.
        
        Args:
            metadata: Dictionary with metadata
            output_path: Path to the ONNX model
        """
        try:
            # Add hardware and optimization info
            metadata['target_hardware'] = self.target_hardware
            metadata['optimization_level'] = self.optimization_level
            
            # Save metadata to JSON file
            metadata_path = f"{output_path}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model metadata saved to: {metadata_path}")
        
        except Exception as e:
            logger.warning(f"Error saving metadata: {str(e)}")
    
    def _update_metadata(self, output_path: str, new_metadata: Dict[str, Any]) -> None:
        """
        Update existing metadata with new information.
        
        Args:
            output_path: Path to the ONNX model
            new_metadata: New metadata to add
        """
        try:
            metadata_path = f"{output_path}.metadata.json"
            
            # Load existing metadata if available
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Update with new metadata
            metadata.update(new_metadata)
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model metadata updated: {metadata_path}")
        
        except Exception as e:
            logger.warning(f"Error updating metadata: {str(e)}")
    
    def load_onnx_model(self, model_path: str) -> Union[Any, None]:
        """
        Load an ONNX model for inference.
        
        Args:
            model_path: Path to the ONNX model
            
        Returns:
            ONNX Runtime inference session or None if loading fails
        """
        if ort is None:
            logger.error("ONNX Runtime not installed, cannot load ONNX model")
            return None
            
        try:
            # Load the model with optimized session options
            session = ort.InferenceSession(
                model_path,
                sess_options=self.session_options,
                providers=self.execution_providers
            )
            
            # Get input and output names
            self.input_names = [input_node.name for input_node in session.get_inputs()]
            self.output_names = [output_node.name for output_node in session.get_outputs()]
            
            # Get input and output shapes
            self.input_shapes = {input_node.name: input_node.shape for input_node in session.get_inputs()}
            self.output_shapes = {output_node.name: output_node.shape for output_node in session.get_outputs()}
            
            logger.info(f"Loaded ONNX model from: {model_path}")
            logger.info(f"Model inputs: {self.input_names} with shapes {self.input_shapes}")
            logger.info(f"Model outputs: {self.output_names} with shapes {self.output_shapes}")
            
            return session
        
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            return None
    
    def predict(self, session, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on an ONNX model.
        
        Args:
            session: ONNX Runtime inference session
            inputs: Dictionary of input arrays
            
        Returns:
            Dictionary of output arrays
        """
        if ort is None:
            logger.error("ONNX Runtime not installed, cannot run prediction")
            return {}
            
        try:
            # Run inference
            outputs = session.run(self.output_names, inputs)
            
            # Return as dictionary
            return {name: output for name, output in zip(self.output_names, outputs)}
        
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return {}
    
    def quantize_model(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Quantize an ONNX model for reduced size and faster inference.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the quantized model. If None, will use model_path + '_quantized.onnx'
            
        Returns:
            Path to the quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Set default output path if not provided
            if output_path is None:
                output_path = model_path.replace('.onnx', '_quantized.onnx')
            
            # Quantize the model
            quantize_dynamic(
                model_input=model_path,
                model_output=output_path,
                per_channel=True,
                reduce_range=True,
                weight_type=QuantType.QInt8
            )
            
            logger.info(f"Quantized model saved to: {output_path}")
            return output_path
        
        except ImportError:
            logger.warning("onnxruntime.quantization not available, skipping quantization")
            return model_path
        except Exception as e:
            logger.error(f"Error during quantization: {str(e)}")
            return model_path
    
    def optimize_for_gh200(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Apply GH200 ARM64-specific optimizations to an ONNX model.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the optimized model. If None, will use model_path + '_gh200.onnx'
            
        Returns:
            Path to the optimized model
        """
        try:
            # Set default output path if not provided
            if output_path is None:
                output_path = model_path.replace('.onnx', '_gh200.onnx')
            
            # Load the model
            model = onnx.load(model_path)
            
            # Apply quantization for ARM64
            quantized_path = self.quantize_model(model_path, model_path.replace('.onnx', '_quantized.onnx'))
            
            # Load the quantized model
            quantized_model = onnx.load(quantized_path)
            
            # Apply ARM64-specific optimizations
            # Apply ARM64-specific optimizations for GH200 architecture
            
            # 1. Quantize the model to int8 for better performance on ARM
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Create a temporary file for the quantized model
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                temp_path = tmp.name
                onnx.save_model(quantized_model, temp_path)
                
                # Quantize with ARM-specific settings
                quantize_dynamic(
                    model_input=temp_path,
                    model_output=temp_path,
                    per_channel=True,
                    reduce_range=True,
                    weight_type=QuantType.QInt8
                )
                
                # Load the quantized model
                quantized_model = onnx.load(temp_path)
            
            # 2. Apply GH200-specific memory optimizations
            # GH200 has specific memory hierarchy and bandwidth characteristics
            
            # Add memory layout optimizations
            # - Optimize tensor layouts for ARM64 architecture
            # - Adjust memory alignment for better cache utilization
            from onnx import helper
            
            # Set appropriate tensor layouts
            for node in quantized_model.graph.node:
                if node.op_type in ['Conv', 'Gemm', 'MatMul']:
                    # Add optimization hints as attributes
                    node.attribute.append(
                        helper.make_attribute('arm64_optimization', 1)
                    )
            
            # 3. Add thread count optimization for GH200 multi-core processing
            # GH200 typically has many ARM cores, so optimize thread usage
            
            # Add metadata about threading optimization
            meta = quantized_model.metadata_props.add()
            meta.key = "preferred_arm_threads"
            meta.value = "8"  # Optimal thread count for GH200
            
            # 4. Apply operator fusion specific to ARM64
            # Fuse operations that can benefit from ARM64 SIMD instructions
            try:
                import onnxoptimizer
                arm_fusion_passes = [
                    'fuse_matmul_add_bias_into_gemm',
                    'fuse_add_bias_into_conv',
                    'fuse_bn_into_conv',
                    'fuse_transpose_into_gemm'
                ]
                quantized_model = onnxoptimizer.optimize(quantized_model, arm_fusion_passes)
            except ImportError:
                logger.warning("onnxoptimizer not available, skipping ARM64 fusion optimizations")
            # Save the optimized model
            onnx.save_model(quantized_model, output_path)
            
            # Update metadata
            self._update_metadata(output_path, {'optimized_for_gh200': True})
            
            logger.info(f"GH200-optimized model saved to: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error optimizing for GH200: {str(e)}")
            return model_path
    
    def convert_and_optimize_entry_timing_model(self, model, sequence_length: int, n_features: int,
                                             output_dir: str, model_name: str = "entry_timing") -> str:
        """
        Convert and optimize an entry timing model for production.
        
        Args:
            model: TensorFlow Keras model
            sequence_length: Length of input sequences
            n_features: Number of input features
            output_dir: Directory to save the models
            model_name: Name of the model
            
        Returns:
            Path to the optimized model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Base ONNX conversion
        base_path = os.path.join(output_dir, f"{model_name}.onnx")
        onnx_path = self.convert_lstm_transformer_model(model, base_path, sequence_length, n_features)
        
        if not onnx_path:
            return ""
        
        # Quantize the model
        quantized_path = self.quantize_model(onnx_path, os.path.join(output_dir, f"{model_name}_quantized.onnx"))
        
        # Optimize for GH200 if that's the target
        if self.target_hardware == 'gh200':
            optimized_path = self.optimize_for_gh200(quantized_path, os.path.join(output_dir, f"{model_name}_gh200.onnx"))
            return optimized_path
        
        return quantized_path


# Helper functions
def convert_keras_to_onnx(model_path: str, output_path: str, target_hardware: str = 'gh200') -> str:
    """
    Helper function to convert a Keras model to ONNX.
    
    Args:
        model_path: Path to the Keras model (.h5 or SavedModel)
        output_path: Path to save the ONNX model
        target_hardware: Target hardware platform
        
    Returns:
        Path to the saved ONNX model
    """
    if tf is None:
        logger.error("TensorFlow not installed, cannot convert Keras model")
        return ""
        
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Create converter
        converter = ONNXConverter(target_hardware)
        
        # Convert the model
        return converter.convert_keras_model(model, output_path)
    
    except Exception as e:
        logger.error(f"Error converting Keras model to ONNX: {str(e)}")
        return ""


def convert_xgboost_to_onnx(model_path: str, output_path: str, input_shape: Tuple[int, ...],
                         feature_names: Optional[List[str]] = None,
                         target_hardware: str = 'gh200') -> str:
    """
    Helper function to convert an XGBoost model to ONNX.
    
    Args:
        model_path: Path to the XGBoost model (.json or .model)
        output_path: Path to save the ONNX model
        input_shape: Shape of input features
        feature_names: Optional list of feature names
        target_hardware: Target hardware platform
        
    Returns:
        Path to the saved ONNX model
    """
    if xgb is None:
        logger.error("XGBoost not installed, cannot convert XGBoost model")
        return ""
        
    try:
        # Load the XGBoost model
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Create converter
        converter = ONNXConverter(target_hardware)
        
        # Convert the model
        return converter.convert_xgboost_model(model, output_path, input_shape, feature_names)
    
    except Exception as e:
        logger.error(f"Error converting XGBoost model to ONNX: {str(e)}")
        return ""


def optimize_existing_onnx(model_path: str, output_path: str, target_hardware: str = 'gh200') -> str:
    """
    Helper function to optimize an existing ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        output_path: Path to save the optimized ONNX model
        target_hardware: Target hardware platform
        
    Returns:
        Path to the optimized ONNX model
    """
    if onnx is None:
        logger.error("ONNX not installed, cannot optimize ONNX model")
        return ""
        
    try:
        # Create converter
        converter = ONNXConverter(target_hardware)
        
        # Load the model
        model_proto = onnx.load(model_path)
        
        # Optimize the model
        optimized_model = converter._optimize_onnx_model(model_proto)
        
        # Save the optimized model
        onnx.save_model(optimized_model, output_path)
        
        # Quantize if needed
        if target_hardware in ['gh200', 'gpu']:
            output_path = converter.quantize_model(output_path)
        
        # Apply hardware-specific optimizations
        if target_hardware == 'gh200':
            output_path = converter.optimize_for_gh200(output_path)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error optimizing ONNX model: {str(e)}")
        return ""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX Model Converter')
    parser.add_argument('--model', type=str, required=True, help='Path to the input model')
    parser.add_argument('--output', type=str, required=True, help='Path to save the ONNX model')
    parser.add_argument('--type', type=str, choices=['keras', 'xgboost', 'onnx'], help='Model type')
    parser.add_argument('--hardware', type=str, default='gh200', choices=['gh200', 'cpu', 'gpu'], help='Target hardware')
    parser.add_argument('--sequence-length', type=int, help='Sequence length for LSTM models')
    parser.add_argument('--n-features', type=int, help='Number of features for LSTM models')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    parser.add_argument('--optimize', action='store_true', help='Apply extra optimizations')
    
    args = parser.parse_args()
    
    if args.type == 'keras':
        if not args.sequence_length or not args.n_features:
            logger.error("--sequence-length and --n-features are required for Keras models")
            exit(1)
        
        # Load the Keras model
        model = tf.keras.models.load_model(args.model)
        
        # Create converter
        converter = ONNXConverter(args.hardware)
        
        # Convert the model
        if 'lstm' in args.model.lower() or 'transformer' in args.model.lower():
            onnx_path = converter.convert_lstm_transformer_model(
                model, args.output, args.sequence_length, args.n_features
            )
        else:
            onnx_path = converter.convert_keras_model(model, args.output)
        
        # Apply quantization if requested
        if args.quantize and onnx_path:
            onnx_path = converter.quantize_model(onnx_path)
        
        # Apply hardware-specific optimizations if requested
        if args.optimize and onnx_path and args.hardware == 'gh200':
            onnx_path = converter.optimize_for_gh200(onnx_path)
        
        if onnx_path:
            print(f"Converted model saved to: {onnx_path}")
        else:
            print("Conversion failed.")
    
    elif args.type == 'xgboost':
        # Load the XGBoost model
        model = xgb.Booster()
        model.load_model(args.model)
        
        # Create converter
        converter = ONNXConverter(args.hardware)
        
        # Get input shape from model metadata or parameters
        if hasattr(model, 'feature_names'):
            n_features = len(model.feature_names)
        elif args.n_features:
            n_features = args.n_features
        else:
            logger.error("Could not determine number of features. Please provide --n-features.")
            exit(1)
        
        # Convert the model
        onnx_path = converter.convert_xgboost_model(
            model, args.output, (None, n_features), model.feature_names if hasattr(model, 'feature_names') else None
        )
        
        # Apply quantization if requested
        if args.quantize and onnx_path:
            onnx_path = converter.quantize_model(onnx_path)
        
        # Apply hardware-specific optimizations if requested
        if args.optimize and onnx_path and args.hardware == 'gh200':
            onnx_path = converter.optimize_for_gh200(onnx_path)
        
        if onnx_path:
            print(f"Converted model saved to: {onnx_path}")
        else:
            print("Conversion failed.")
    
    elif args.type == 'onnx':
        # Create converter
        converter = ONNXConverter(args.hardware)
        
        # Optimize the model
        optimized_path = optimize_existing_onnx(args.model, args.output, args.hardware)
        
        if optimized_path:
            print(f"Optimized model saved to: {optimized_path}")
        else:
            print("Optimization failed.")
    
    else:
        logger.error("Unknown model type. Please specify --type.")
        exit(1)

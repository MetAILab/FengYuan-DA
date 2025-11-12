import onnxruntime as ort
import numpy as np
import torch
import torch.nn.functional as F
import os
from typing import Tuple, Dict, Any

class ONNXInference:
    def __init__(self, onnx_model_path: str, device: str = 'cuda'):
        """
        Initialize ONNX inference session
        
        Args:
            onnx_model_path: Path to the ONNX model file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.onnx_model_path = onnx_model_path
        self.device = device
        
        # Create inference session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Get input and output details
        self.input_details = {input.name: input for input in self.session.get_inputs()}
        self.output_details = {output.name: output for output in self.session.get_outputs()}
        
        print("ONNX model loaded successfully!")
        print("Inputs:", list(self.input_details.keys()))
        print("Outputs:", list(self.output_details.keys()))
    
    def preprocess_data(self, background: np.ndarray, observation: np.ndarray, 
                       target_size: Tuple[int, int] = (721, 1440)) -> Dict[str, np.ndarray]:
        """
        Preprocess input data similar to your original test function
        
        Args:
            background: Background data array
            observation: Observation data array  
            target_size: Target size for interpolation (height, width)
        
        Returns:
            Dictionary of preprocessed inputs for ONNX model
        """
        # Ensure data is float32
        background = background.astype(np.float32)
        observation = observation.astype(np.float32)
        
        # Add batch dimension if not present
        if background.ndim == 3:
            background = background[np.newaxis, ...]  # (1, 69, 721, 1440)
        if observation.ndim == 3:
            observation = observation[np.newaxis, ...]  # (1, 69, 721, 1440)
        
        # Resize observation if needed (similar to your F.interpolate)
        if observation.shape[-2:] != target_size:
            # Use simple resize - in practice you might want more sophisticated interpolation
            from scipy import ndimage
            obs_resized = np.zeros((observation.shape[0], observation.shape[1], target_size[0], target_size[1]), dtype=np.float32)
            for b in range(observation.shape[0]):
                for c in range(observation.shape[1]):
                    obs_resized[b, c] = ndimage.zoom(observation[b, c], 
                                                    (target_size[0]/observation.shape[2], 
                                                     target_size[1]/observation.shape[3]), 
                                                    order=1)  # bilinear interpolation
            observation = obs_resized
        
        return {
            'background': background,
            'observation': observation
        }
    
    def infer(self, background: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """
        Run inference using ONNX model
        
        Args:
            background: Background data array
            observation: Observation data array
            
        Returns:
            Analysis result
        """
        # Preprocess data
        inputs = self.preprocess_data(background, observation)
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        return outputs[0]  # Return first output
    

# Usage example
def main():
    # Initialize ONNX inference
    onnx_model_path = "./Fengyuan_DA.onnx"
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    inference_engine = ONNXInference(onnx_model_path, device='cuda')
    
    # Example with single inference
    print("\n=== Single Inference Example ===")
    
    # Create dummy data (replace with your actual data)
    background = np.random.randn(1, 69, 721, 1440).astype(np.float32)
    observation = np.random.randn(1, 69, 721, 1440).astype(np.float32)
    
    # Run inference
    analysis = inference_engine.infer(background, observation)
    analysis = analysis * std + mean
    print(f"Analysis shape: {analysis.shape}")
    
    # Example with data loader (similar to your test function)
    print("\n=== Batch Inference Example ===")
    


if __name__ == "__main__":
    main()
# FengYuan-DA
FengYuan Data Assimilation Inference providing efficient inference using ONNX models. This module applies deep learning models to data assimilation tasks, combining background fields and observation data to generate more accurate analysis fields.
## Background: Fengyuan Data Assimilation

- **Combine data from different sources**: Fuse background fields from numerical weather prediction models with actual observation data
- **Generate analysis fields**: Produce optimized initial conditions for weather forecasting models

## Installation

```bash
pip install onnxruntime numpy torch
```

## Quick Start

```python

# Initialize the inference engine
inference_engine = ONNXInference(
    onnx_model_path="path/to/your/model.onnx",
    device='cuda'  # Use 'cpu' for CPU-only environments
)

# Prepare your data
background = np.random.randn(1, 69, 721, 1440).astype(np.float32)  # Background field
observation = np.random.randn(1, 69, 721, 1440).astype(np.float32)  # Observation data

# Run data assimilation
analysis = inference_engine.infer(background, observation)
print(f"Analysis field shape: {analysis.shape}")
```

## Input Data Specifications

### Background Field
- **Shape**: `(batch_size, 69, 721, 1440)`
- **Type**: `float32`
- **Description**: Numerical weather prediction background field
- **Dimensions**: (vertical levels, latitude, longitude)
- **Variable Order**：The data contains 69 variables: 4 surface variables and 65 pressure level variables across 13 standard levels (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa). The variables are arranged in the following order:
   Surface: u10, v10, t2m, msl
   Geopotential height (Z) for all 13 levels
   Specific humidity (q) for all 13 levels
   U-component of wind (U) for all 13 levels
   V-component of wind (V) for all 13 levels
   Temperature (T) for all 13 levels
### Observation Data
- **Shape**: `(batch_size, 69, 721, 1440)`
- **Type**: `float32`
- **Description**: Actual measurement data from various sources
- **Variable Order**：The data contains 69 variables: 4 surface variables and 65 pressure level variables across 13 standard levels (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa). The variables are arranged in the following order:
   Surface: u10, v10, t2m, msl
   Geopotential height (Z) for all 13 levels
   Specific humidity (q) for all 13 levels
   U-component of wind (U) for all 13 levels
   V-component of wind (V) for all 13 levels
   Temperature (T) for all 13 levels
 
### Output Analysis
- **Shape**: `(batch_size, 69, 721, 1440)`
- **Type**: `float32`
- **Description**: Optimized analysis field combining background and observations

## Advanced Usage

### Batch Processing
```python
# Process multiple time steps efficiently
backgrounds = np.random.randn(1, 69, 721, 1440).astype(np.float32)
observations = np.random.randn(1, 69, 721, 1440).astype(np.float32)

results = []
for bg, obs in zip(backgrounds, observations):
    analysis = inference_engine.infer(bg, obs)
    results.append(analysis)
```

### Custom Data Preprocessing
```python
# Extend preprocessing for specific data formats
class CustomInference(ONNXInference):
    def preprocess_data(self, background, observation, target_size=(721, 1440)):
        # Add custom normalization or filtering
        background = self.custom_normalize(background)
        observation = self.filter_observations(observation)
        return super().preprocess_data(background, observation, target_size)
```

## Model Requirements

The ONNX model should:
- Accept two inputs: `background` and `observation`
- Output a single analysis field
```

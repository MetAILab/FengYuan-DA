# FengYuan-DA
FengYuan Data Assimilation Model providing efficient inference using ONNX models. 

## Installation

```bash
pip install onnxruntime numpy torch
```
## Model Download
The pre-trained ONNX model can be downloaded from:
ONNX Model Drive Download:
https://

## Usage
Run the inference_DA script directly:
```bash
python inference_DA.py
```
  
## Data Assimilation Process
The FengYuan-DA model performs data assimilation by combining:
- 6-hour forecast field from Fengyuan NWP model (background)
- Gridded observational data from GDAS system (observation)
Workflow Example:
- Input: 2022010100 6-hour forecast (background) + 2022010106 observations
- Output: 2022010106 analysis field (initial condition)
The generated analysis field serves as the initial condition for subsequent numerical weather prediction runs.

## Input Data Specifications

### Background Field
- **Shape**: `(batch_size, 69, 721, 1440)`
- **Type**: `float32`
- **Description**: 6-hour forecast data from the Fengyuan numerical weather prediction model. This represents the model's prior estimate of atmospheric states before assimilation.
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
- **Description**: Gridded observational data derived from GDAS (Global Data Assimilation System) observations. 
- **Variable Order**：The data contains 69 variables: 4 surface variables and 65 pressure level variables across 13 standard levels (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa). The variables are arranged in the following order:
   Surface: u10, v10, t2m, msl
   Geopotential height (Z) for all 13 levels
   Specific humidity (q) for all 13 levels
   U-component of wind (U) for all 13 levels
   V-component of wind (V) for all 13 levels
   Temperature (T) for all 13 levels
 
### Output Analysis
**Shape**:(batch_size, 69, 721, 1440)
**Type**: float32
**Description**: Optimized analysis field combining background and observations. This analysis serves as the initial condition for subsequent numerical weather prediction forecasts.

## Model Requirements

The ONNX model should:
- Accept two inputs: `background` and `observation`
- Output a single analysis field
```

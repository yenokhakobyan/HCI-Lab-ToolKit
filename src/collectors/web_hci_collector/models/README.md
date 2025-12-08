# Model Files

Place pre-trained models here for cognitive state detection.

## Recommended Model: DenseAttNet

For learning-specific cognitive state detection (confusion, engagement, boredom, frustration):

1. **Download or train a DAiSEE model**
   - Dataset: https://iith.ac.in/~daisee-dataset/
   - Paper: "3D DenseAttNet for Student Engagement Recognition"

2. **Convert to ONNX format** (if using PyTorch):
   ```python
   import torch

   model = torch.load('denseattnet.pt')
   model.eval()

   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(
       model,
       dummy_input,
       'denseattnet.onnx',
       input_names=['input'],
       output_names=['output'],
       dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
   )
   ```

3. **Place the file here as**: `denseattnet.onnx`

## Expected Model Output

The model should output 4 values (one per state):
- Index 0: Engagement (0-1)
- Index 1: Boredom (0-1)
- Index 2: Confusion (0-1)
- Index 3: Frustration (0-1)

## Demo Mode

Without a model file, the system runs in demo mode with simulated predictions.
This is useful for testing the UI and data pipeline.

## Alternative Models

Other compatible models:
- ResNet+TCN (63.9% accuracy on DAiSEE)
- EfficientNetV2-L+LSTM (62.1% accuracy)
- Any model trained on DAiSEE with 4-class output

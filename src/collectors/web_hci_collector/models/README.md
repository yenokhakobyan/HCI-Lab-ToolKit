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

## Default Mode (Landmark-Based Heuristics)

Without a model file, the system estimates cognitive states from MediaPipe
face mesh landmarks using FACS-based heuristic indicators:
- Eye Aspect Ratio (EAR) for drowsiness/engagement
- Eyebrow position for confusion/frustration
- Head pose stability for engagement
- Mouth aspect ratio for boredom (yawning detection)
- Blink rate for frustration detection

Confidence values are lower (0.45-0.70) compared to model-based detection.
For higher accuracy, install a pre-trained model as described above.

## Alternative Models

Other compatible models:
- ResNet+TCN (63.9% accuracy on DAiSEE)
- EfficientNetV2-L+LSTM (62.1% accuracy)
- Any model trained on DAiSEE with 4-class output

import onnx
import torch
import torch.onnx

model = torch.load("models/posevae_c1_e6_l32.pt")
input_shape = (32, 31379)

torch.onnx.export(model,
                  (torch.randn(input_shape), torch.randn(input_shape)),
                  "models/posevae_onnx.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  verbose=2,
                  opset_version=11)

# onnx_model = onnx.load("models/posevae_onnx.onnx")
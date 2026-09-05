import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

import numpy as np
import onnxruntime as ort

def build_initial_model(num_classes: int) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.DEFAULT

    model = mobilenet_v3_large(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier[0].parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model

checkpoint = torch.load("models/mobilenet_v3_large_nui.pth")
num_classes = len(checkpoint["class_to_idx"])

model = build_initial_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

Path("web/public/models").mkdir(parents=True, exist_ok=True)
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "web/public/models/mobilenet_v3_large_nui.onnx",
    input_names=["image"],
    output_names=["logits"],
    # Prevent generating 2 files(.onnx and .data) that the frontend can't recognize
    dynamo=False,
)

class_to_idx = checkpoint["class_to_idx"]
class_labels = [None] * len(class_to_idx)
for label, idx in class_to_idx.items():
    class_labels[idx] = label

with open("web/public/models/class_to_idx.json", "w") as f:
    json.dump(class_labels, f)

def verify_onnx_export(
    model: nn.Module, onnx_path: str, dummy_input: torch.Tensor
) -> None:
    """PyTorchモデルとONNXモデルが同じ入力に対して同じ出力を返すか検証する"""
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(
        None,  # Noneを渡すとexport時に指定した全ての出力（今回は"logits"のみ）を返す
        {"image": dummy_input.numpy()},
    )[0]

    if not np.allclose(pytorch_output, onnx_output, atol=1e-4):
        max_diff = np.abs(pytorch_output - onnx_output).max()
        raise ValueError(
            f"PyTorchとONNXの出力が一致しません（最大差分: {max_diff}）"
        )

    print("OK: PyTorchとONNXの出力は一致しています")

verify_onnx_export(model, "web/public/models/mobilenet_v3_large_nui.onnx", dummy_input)

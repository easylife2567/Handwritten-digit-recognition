import os
import torch
from torch import nn
from train import MLP, get_device

def export_onnx(pt_path: str = "best.pt", onnx_path: str = "web/public/mnist_dnn.onnx"):
    device = get_device()
    model = MLP().to(device)
    state = torch.load(pt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, 28, 28, device=device)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()

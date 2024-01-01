import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

class CosSimilarity(nn.Module):
    def __init__(self):
        super(CosSimilarity, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        x1 and x2 are normalized features
        """
        similarity = torch.matmul(x1, x2.transpose(0, 1))
        similarity = torch.maximum(torch.tensor(0.0), similarity)
        return similarity

model = CosSimilarity()
model.eval()
model.cpu()

MODEL_NAME = "cos_similarity"
RESOLUTION = [
    [1,2048],
]

for N, F in RESOLUTION:
    onnx_file = f"{MODEL_NAME}_11x{F}.onnx"
    x = torch.randn(N, F).cpu()
    y = torch.randn(N, F).cpu()
    torch.onnx.export(
        model,
        args=(x,y),
        f=onnx_file,
        opset_version=11,
        input_names=['feature1','feature2'],
        output_names=['similarity'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)


    onnx_file = f"{MODEL_NAME}_1Nx{F}.onnx"
    x = torch.randn(N, F).cpu()
    y = torch.randn(N, F).cpu()
    torch.onnx.export(
        model,
        args=(x,y),
        f=onnx_file,
        opset_version=11,
        input_names=['feature1','features2'],
        output_names=['similarities'],
        dynamic_axes={
            'features2' : {0: 'N'},
            'similarities' : {0: '1', 1: 'N'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)


    onnx_file = f"{MODEL_NAME}_NMx{F}.onnx"
    x = torch.randn(N, F).cpu()
    y = torch.randn(N, F).cpu()
    torch.onnx.export(
        model,
        args=(x,y),
        f=onnx_file,
        opset_version=11,
        input_names=['features1','features2'],
        output_names=['similarities'],
        dynamic_axes={
            'features1' : {0: 'N'},
            'features2' : {0: 'M'},
            'similarities' : {0: 'N', 1: 'M'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

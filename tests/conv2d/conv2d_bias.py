import torch
from converter.converter import parse_onnx_model, dump_onnx_model
from tests.helpers import write_bin, onnx_check_model
import torch.onnx
import onnx
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    filename = __file__.split('/')[-1][:-3]
    x = torch.randn(1, 2, 5, 5)
    model = torch.nn.Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=True)
    
    model = model.cpu()
    x = x.cpu()
    out = model(x)

    write_bin(args.output + "/" + filename + "_input.bin", x.numpy())

    torch.onnx.export(model,
                    x,
                    args.output + "/" + "conv2d_bias.onnx",
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names = ['input'],
                    output_names = ['output']
                    )

    model_onnx = onnx.load(args.output + "/" + "conv2d_bias.onnx")
    onnx_check_model(model_onnx, args.output + "/" + "conv2d_bias")
    ir = parse_onnx_model(model_onnx)
    dump_onnx_model(ir, args.output + "/" + filename + "_ir.bin", verbose=False)
    write_bin(args.output + "/" + filename + "_output.bin", out.detach().numpy())
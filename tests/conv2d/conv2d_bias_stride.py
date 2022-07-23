import torch
from converter.converter import parse_onnx_model, dump_onnx_model
from tests.helpers import write_bin, onnx_check_model, seed_everything
import torch.onnx
import onnx
import argparse

if __name__ == "__main__":
    seed_everything(seed=42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    batch = 2
    height, width = 4, 4
    in_channels = 2 
    out_channels = 3
    kernel_size = 2
    stride = 2
    padding = 0

    filename = __file__.split('/')[-1][:-3]

    x = torch.arange(batch*in_channels*height*width, dtype=torch.float).reshape(batch, in_channels, height, width)
    model = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, dtype=torch.float)
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
    write_bin(args.output + "/" + filename + "_output_py.bin", out.detach().numpy())

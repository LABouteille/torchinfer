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
    x = torch.arange(3*2*4*4, dtype=torch.float).reshape(3, 2, 4, 4)
    # Need to put hypera-meters as constant. I can put 999 because right after i'm overwritting
    # de weights
    model = torch.nn.Conv2d(2, 999, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=True, dtype=torch.float)
    model.weight.data = torch.arange(2*2*2*2, dtype=torch.float).reshape(2, 2, 2, 2)
    model.bias.data = torch.zeros((2,), dtype=torch.float)
    model = model.cpu()
    x = x.cpu()
    out = model(x)
    # print(x, x.shape)
    # print("======")
    # print(model.weight.data, model.weight.data.shape)
    # print(model.bias.data, model.bias.data.shape)
    # print("======")
    # print(out, out.shape)

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

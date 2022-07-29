import struct
import numpy as np
import onnx
import os
import random
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def write_bin(filename, array):
    # Force endianess: https://stackoverflow.com/questions/23831422/what-endianness-does-python-use-to-write-into-files
    dtype_to_format = {
        np.int: 'i',
        np.int8: 'i',
        np.int16: 'i',
        np.int32: 'i',
        np.int64: 'i',
        np.unsignedinteger: 'I',
        np.float: 'f',
        np.float16: 'f',
        np.float32: 'f',
        np.float64: 'f',
        np.double: 'd'
    }
    fmt = dtype_to_format[array.dtype.type]
    n, c, h, w = array.shape
    with open(filename, "wb") as f:
        f.write(struct.pack('I', n))
        f.write(struct.pack('I', c))
        f.write(struct.pack('I', h))
        f.write(struct.pack('I', w))
        f.write(struct.pack('c', bytes(fmt, 'utf-8')))
        f.write(struct.pack(f"{fmt}"*(n*c*h*w), *array.flatten(order="C").tolist()))

def read_bin(filename):
    # https://qiita.com/madaikiteruyo/items/dadc99aa29f7eae0cdd0
    format_to_byte = {
        'c': 1,
        'i': 4,
        'I': 4,
        'f': 4,
        'd': 8
    }

    data = []
    dims, fmt = None, None
    with open(filename, "rb") as f:
        # read row and col (np.int = 4 bytes)
        byte = f.read(4 * format_to_byte['i'])
        if byte == b'':
            raise Exception("read_bin: Empty binary")
        else:
            dims = struct.unpack('IIII', byte)
        
        if len(dims) != 4: raise Exception("read_bin: No dimensions (n,c,h,w) dumped in binary")

        # Read character format
        byte = f.read(1)
        if byte == b'':
            raise Exception("read_bin: Empty binary")
        else:
            fmt = chr(struct.unpack('c', byte)[0][0])

        if len(fmt) != 1: raise Exception("read_bin: No format dumped in binary")
        
        while True:
            byte = f.read(format_to_byte[fmt])
            if byte == b'':
                break
            else:
                data.append(struct.unpack(fmt, byte)[0])
        
    return np.array(data).reshape(dims[0], dims[1], dims[2], dims[3])

def onnx_check_model(model_onnx, name):
    try:
        onnx.checker.check_model(model_onnx)
    except onnx.checker.ValidationError as e:
        raise Exception(f"{name} is invalid: %s" % e)
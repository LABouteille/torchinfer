from collections import OrderedDict
from enum import IntEnum
from typing import Dict, Optional, List
import onnx

class TYPEDEF:
    ONNX_MODEL = onnx.onnx_ml_pb2.ModelProto
    ONNX_NODE = onnx.onnx_ml_pb2.NodeProto
    ONNX_IR = OrderedDict

class OPTYPE(IntEnum):
    CONV2D = 0
    RELU = 1
    FLATTEN = 2
    LINEAR = 3

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name

class DTYPE(IntEnum):
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto#L483-L485
    FLOAT = 1,   # float
    INT8 =  3,   # int8_t
    INT16 = 4,   # int16_t
    INT32 = 6,   # int32_t
    INT64 = 7    # int64_t

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name

def dump_onnx_model(ir: TYPEDEF.ONNX_IR) -> None:
    pass

def get_dtype(data_type: int) -> DTYPE:
    if data_type == DTYPE.FLOAT:
        return DTYPE.FLOAT
    elif data_type == DTYPE.INT8:
        return DTYPE.INT8
    elif data_type == DTYPE.INT16:
        return DTYPE.INT16
    elif data_type == DTYPE.INT32:
        return DTYPE.INT32
    else:
        return DTYPE.INT64

def get_optype(node: TYPEDEF.ONNX_NODE) -> OPTYPE:
    optype = node.op_type.lower()

    if optype == "conv":
        return OPTYPE.CONV2D
    elif optype == "relu":
        return OPTYPE.RELU
    elif optype == "gemm":
        return OPTYPE.LINEAR
    elif optype == "flatten":
        return OPTYPE.FLATTEN
    else:
        raise NotImplementedError("get_optype: Operator type not implemented yet.")    

def get_attributes(node: TYPEDEF.ONNX_NODE) -> OrderedDict:
    raise NotImplementedError("get_attributes")

def get_initializer(node: TYPEDEF.ONNX_NODE, model: TYPEDEF.ONNX_MODEL) -> Dict[Dict, Optional[Dict]]:
    initializer = dict()
    initializer["weight"], initializer["bias"] = {}, {}
    
    for inp in node.input:
        for init in model.graph.initializer:
            if inp == init.name:
                #FIXME: What happens if no bias ?
                key = "weight" if len(init.dims) > 1 else "bias"
                initializer[key]["name"] = init.name
                initializer[key]["raw_data"] = init.raw_data
                initializer[key]["dims"] = list(init.dims)
                initializer[key]["data_type"] = get_dtype(init.data_type)

    return initializer


def get_nodes_input_output(ir: OrderedDict, model: TYPEDEF.ONNX_MODEL) -> None:        
    nb_nodes = len(model.graph.node)
    prev_node = None

    for i in range(nb_nodes):
        node = model.graph.node[i]
        
        if i + 1 < nb_nodes:
            next_node = model.graph.node[i+1]

        if i != 0:
            ir[node.name]["inputs"] = [prev_node.name]
        if i != nb_nodes - 1:
            ir[node.name]["outputs"] = [next_node.name]
            prev_node = node

    # Input of first node
    ir[model.graph.node[0].name]["inputs"] = [inp.name for inp in model.graph.input] 

    # Output of last node
    ir[model.graph.node[-1].name]["outputs"] = [out.name for out in model.graph.output] 

def get_input_nodes(model: TYPEDEF.ONNX_MODEL) -> List:
    return [node.name for node in model.graph.input]

def get_output_nodes(model: TYPEDEF.ONNX_MODEL) -> List:
    return [node.name for node in model.graph.output]
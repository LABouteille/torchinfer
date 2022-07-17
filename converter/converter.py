from converter.helpers import *

import struct

def parse_onnx_model(model: TYPEDEF.ONNX_MODEL) -> OrderedDict:
    """
    OrderedDict
    {
        node_ith: {
            op_type: OPTYPE
            name: str 
            inputs: 
            outputs: List
            initializer: Dict
            attributes: {}
            data_layout: Union(NCHW, NHWC)
        }
    }

    Returns an intermediate representation of ONNX model
    """
    ir = OrderedDict()

    # TODO: Need to handle input 
    ir[0] = get_input_nodes(model)

    for idx, node in enumerate(model.graph.node, start=1):
        ir[idx] = {}
        ir[idx]["name"] = node.name
        ir[idx]["op_type"] = get_optype(node)
        ir[idx]["initializer"] = get_initializer(node, model)
        ir[idx]["attributes"] = None
        ir[idx]["data_layout"] = None

    # Get inputs and outputs of each intermediate nodes.
    get_nodes_input_output(ir, model)        
    
    # TODO: Need to handle output
    # ir[len(model.graph.node) + 1] = get_output_nodes(model)

    return ir

def dump_onnx_model(ir: TYPEDEF.ONNX_IR, filename, verbose=False) -> None:
    """
        - Nb_layer
        for all layers:
            - layer_id
            - name size
            - name
            - op_type
            if Input:
                - dims                
            if Conv2d:
                - nb_params
                - dims (weight)
                - weight
                - dims (bias)
                - bias
    """

    with open(filename, "wb") as f:
        # Nb layers
        f.write(struct.pack('i', len(ir)))

        for i, (idx, layer) in enumerate(ir.items()):
            if verbose: print(idx, layer["name"], layer["inputs"], layer["outputs"])

            # Layer id
            f.write(struct.pack('i', idx))
            # Name size
            f.write(struct.pack('i', len(layer["name"])))
            # Name
            f.write(struct.pack(f"{len(layer['name'])}s", str.encode(layer["name"])))

            if layer["op_type"] == OPTYPE.INPUT:
                # Op type
                f.write(struct.pack('i', layer["op_type"]))
                # Dims
                f.write(struct.pack("i"*len(layer["dims"]), *layer["dims"]))
            elif layer["op_type"] == OPTYPE.CONV2D:
                # TODO: Need to write idx of input and output layer.
                # Op type
                f.write(struct.pack('i', layer["op_type"]))
                # Nb params
                f.write(struct.pack('i', len(layer["initializer"])))
                # Weight
                f.write(struct.pack("i"*len(layer["initializer"]["weight"]["dims"]), *layer["initializer"]["weight"]["dims"]))
                weight = layer["initializer"]["weight"]["raw_data"] 
                f.write(struct.pack(f"{len(weight)}s", weight)) # len = data_type * nb elements
                # Bias
                if "bias" in layer["initializer"]:
                    f.write(struct.pack("i"*len(layer["initializer"]["bias"]["dims"]), *layer["initializer"]["bias"]["dims"]))
                    bias = layer["initializer"]["bias"]["raw_data"]
                    f.write(struct.pack(f"{len(bias)}s", bias))
            else:
                raise NotImplementedError("dump_onnx_model: Not implemented layer")

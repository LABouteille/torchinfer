from helpers import *

def parse_onnx_model(model: TYPEDEF.ONNX_MODEL) -> OrderedDict:
    """
    OrderedDict
    {
        input_nodes: []

        node_ith: {
            op_type: OPTYPE
            name: str 
            inputs: 
            outputs: List
            initializer: Dict
            attributes: {}
            data_layout: Union(NCHW, NHWC)
        }

        output_nodes: []
    }

    Returns an intermediate representation of ONNX model
    """
    ir = OrderedDict()

    ir["input_nodes"] = get_input_nodes(model)
    
    for idx, node in enumerate(model.graph.node):
        ir[node.name] = {}
        ir[node.name]["op_type"] = get_optype(node)
        ir[node.name]["initializer"] = get_initializer(node, model)
        ir[node.name]["attributes"] = None
        ir[node.name]["data_layout"] = None

    get_nodes_input_output(ir, model)        

    ir["output_nodes"] = get_output_nodes(model)

    return ir
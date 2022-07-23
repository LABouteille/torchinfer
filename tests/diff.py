from tests.helpers import read_bin
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", type=str, required=True, help="Binary file from Python")
    parser.add_argument("--cpp", type=str, required=True, help="Binary file from C++")
    args = parser.parse_args()

    array_py = read_bin(args.py)
    array_cpp = read_bin(args.cpp)

    if np.allclose(array_py, array_cpp):
        exit(0)
    else:
        exit(1)
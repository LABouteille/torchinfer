#!/bin/bash

RED="\e[31m"
GREEN="\e[32m"
ORANGE="\e[33m"
END_COLOR="\e[0m"

BUILD_PATH=$PWD/build
TESTS_PATH=$PWD/tests
TESTS_TMP_PATH=${TESTS_PATH}/tmp

IS_CI=0 # Only when runing tests on github action CI/CD
IS_VERBOSE=0
IS_FILTER=0
IS_GDB=0
REGEX_CMD=""

while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --is_ci)
            IS_CI=1
            # For CI/CD github actions only
            shift # past argument
            ;;
        --gdb)
            IS_GDB=1
            # Run binary with GDB debugger
            shift # past argument
            ;;
        --verbose)
            IS_VERBOSE=1
            # Enable logging
            shift # past argument
            ;;
        --filter=*)
            IS_FILTER=1
            # Filter tests by regex
            REGEX_CMD="${key#*=}"
            shift # past argument
            ;;
        --help)
            # Help menu
            echo "Usage: run_tests.sh [--is_ci] [--gdb] [--verbose] [--filter=<regex>] [--help]"
            exit 0
            ;;
        *)    # unknown option
            shift # past argument
            ;;
    esac
done

if [ $IS_CI -eq 0 ]
then

    if [ -d ${TESTS_TMP_PATH} ] 
    then
        rm -rf ${TESTS_TMP_PATH}
    fi

    mkdir ${TESTS_TMP_PATH}

    cd ${BUILD_PATH}
    cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j 16
    cd ..
    cp ${BUILD_PATH}/targets/torchinfer ${TESTS_TMP_PATH}
fi

if [ $IS_CI -eq 0 ]
then
    source ./torchinfer-env/bin/activate
fi

mkdir -p ${TESTS_TMP_PATH}/conv2d
conv2d_list=($(find ${TESTS_TMP_PATH}/../conv2d -name "*.py"))

#Keep only line from lists that match REGEX_CMD
if [ $IS_FILTER -eq 1 ]
then
    for i in "${!conv2d_list[@]}"
    do
        if [[ ! ${conv2d_list[$i]} =~ $REGEX_CMD ]]
        then
            unset "conv2d_list[$i]"
        fi
    done
fi
# TODO: add GDB flags

counter=0

for filename in "${conv2d_list[@]}"
do
    testname=${filename#${TESTS_TMP_PATH}/../}
    testname=${testname%.py}
    python ${filename} --output ${TESTS_TMP_PATH}/conv2d

    if [ $IS_CI -eq 0 ]
    then
        VERBOSE=""

        if [ $IS_VERBOSE -eq 1 ]
        then
            VERBOSE="--verbose"
        fi
        
        if [ $IS_GDB -eq 1 ]
        then
            gdb --args ${TESTS_TMP_PATH}/torchinfer ${VERBOSE} --input ${TESTS_TMP_PATH}/${testname}_input.bin --type float --onnx_ir ${TESTS_TMP_PATH}/${testname}_ir.bin --output ${TESTS_TMP_PATH}/${testname}_output_cpp.bin
        else
            ${TESTS_TMP_PATH}/torchinfer ${VERBOSE} --input ${TESTS_TMP_PATH}/${testname}_input.bin --type float --onnx_ir ${TESTS_TMP_PATH}/${testname}_ir.bin --output ${TESTS_TMP_PATH}/${testname}_output_cpp.bin
        fi

    else
        ${BUILD_PATH}/targets/torchinfer --input ${TESTS_TMP_PATH}/${testname}_input.bin --type float --onnx_ir ${TESTS_TMP_PATH}/${testname}_ir.bin --output ${TESTS_TMP_PATH}/${testname}_output_cpp.bin
    fi
    
    python ${TESTS_PATH}/diff.py --py ${TESTS_TMP_PATH}/${testname}_output_py.bin --cpp ${TESTS_TMP_PATH}/${testname}_output_cpp.bin

    if [ $? -eq 0 ]
    then
        echo -e "${GREEN}Test \"${testname}\" passed${END_COLOR}"
        counter=$((counter+1))
    else
        echo -e "${RED}Test \"${testname}\" failed${END_COLOR}"
    fi
done

nb_tests=${#conv2d_list[@]}

if [ $counter -eq $nb_tests ]
then
    echo -e "${GREEN}${counter}/${nb_tests} tests passed${END_COLOR}"
    exit 0
else
    echo -e "${ORANGE}${counter}/${nb_tests} tests passed${END_COLOR}"
    exit 1
fi
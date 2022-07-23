#!/bin/bash

# Only when runing tests on github action CI/CD
if [ "$1" == "--is_ci" ]; then
    is_ci=1
else
    is_ci=0
fi

RED="\e[31m"
GREEN="\e[32m"
ORANGE="\e[33m"
END_COLOR="\e[0m"

BUILD_PATH=$PWD/build
TESTS_PATH=$PWD/tests
TESTS_TMP_PATH=${TESTS_PATH}/tmp

if [ $is_ci -eq 0 ]
then

    if [ -d ${TESTS_TMP_PATH} ] 
    then
        rm -rf ${TESTS_TMP_PATH}
    fi

    mkdir ${TESTS_TMP_PATH}

    if [ -d ${BUILD_PATH} ] 
    then
        rm -rf ${BUILD_PATH}
    fi

    mkdir ${BUILD_PATH}

    # TODO: add regex as cmd line arg to filter tests per layers (one particuliar test or all test of a specific layer)
    # TODO: Upload the build directory as artefact and reuse across CI jobs.
    cd ${BUILD_PATH}
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j 16
    cd ..
    cp ${BUILD_PATH}/targets/torchinfer ${TESTS_TMP_PATH}
fi

if [ $is_ci -eq 0 ]
then
    source ./torchinfer-env/bin/activate
fi

mkdir -p ${TESTS_TMP_PATH}/conv2d
conv2d_list=($(find ${TESTS_TMP_PATH}/../conv2d -name "*.py"))

counter=0

for ((i=0; i<${#conv2d_list[@]}; i++));
do
    filename=${conv2d_list[$i]} 
    testname=${filename#${TESTS_TMP_PATH}/../}
    testname=${testname%.py}

    python ${filename} --output ${TESTS_TMP_PATH}/conv2d

    if [ $is_ci -eq 0 ]
    then
        ${TESTS_TMP_PATH}/torchinfer --input ${TESTS_TMP_PATH}/${testname}_input.bin --type float --onnx_ir ${TESTS_TMP_PATH}/${testname}_ir.bin --output ${TESTS_TMP_PATH}/${testname}_output_cpp.bin
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
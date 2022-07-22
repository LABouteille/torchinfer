#!/bin/bash

RED="\e[31m"
GREEN="\e[32m"
ORANGE="\e[33m"
END_COLOR="\e[0m"

BUILD_PATH=$PWD/../build
TESTS_TMP_PATH=$PWD/tmp

if [ -d ${TESTS_TMP_PATH} ] 
then
    rm -rf ${TESTS_TMP_PATH}
fi

mkdir ${TESTS_TMP_PATH}

# if [ -d ${BUILD_PATH} ] 
# then
#     rm -rf ${BUILD_PATH}
# fi

# TODO: Upload the build directory as artefact and reuse across CI jobs.
cmake -B ${BUILD_PATH} .. -DCMAKE_BUILD_TYPE=Release && make -j 16 -C ${BUILD_PATH}
cp ${BUILD_PATH}/targets/torchinfer ${TESTS_TMP_PATH}
cd ../

source torchinfer-env/bin/activate

mkdir -p ${TESTS_TMP_PATH}/conv2d
conv2d_list=($(find ${TESTS_TMP_PATH}/../conv2d -name "*.py"))

counter=0

for ((i=0; i<${#conv2d_list[@]}; i++));
do
    filename=${conv2d_list[$i]} 
    testname=${filename#${TESTS_TMP_PATH}/../}
    testname=${testname%.py}

    python ${filename} --output ${TESTS_TMP_PATH}/conv2d

    ${TESTS_TMP_PATH}/torchinfer --input ${TESTS_TMP_PATH}/${testname}_input.bin --type float --onnx_ir ${TESTS_TMP_PATH}/${testname}_ir.bin --output ${TESTS_TMP_PATH}/${testname}_output_cpp.bin
    
    diff ${TESTS_TMP_PATH}/${testname}_output_py.bin ${TESTS_TMP_PATH}/${testname}_output_cpp.bin

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
else
    echo -e "${ORANGE}${counter}/${nb_tests} tests passed${END_COLOR}"
fi
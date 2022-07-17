#!/bin/bash

source ../torchinfer-env/bin/activate

if [ -d "tmp/" ] 
then
    rm -rf tmp/ 
fi

mkdir tmp/

conv2d_list=($(find conv2d -name "*.py"))
counter=0

for ((i=0; i<${#conv2d_list[@]}; i++));
do
    python ${conv2d_list[i]} --output tmp
    counter=$((counter+1))
done

nb_tests=${#conv2d_list[@]}
echo "${counter}/${nb_tests} tests passed"
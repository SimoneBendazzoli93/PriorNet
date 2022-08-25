#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
TESTPATH=${INPUT_PATH}
TESTOUTPUTPATH=${OUTPUT_PATH}
./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="16g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create priornet-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $TESTPATH:/input/ \
        -v priornet-output-$VOLUME_SUFFIX:/output/ \
        priornet

docker run --rm -it \
        -v priornet-output-$VOLUME_SUFFIX:/output/ \
        -v $TESTOUTPUTPATH:/expected_output/ \
        priornet python -c """
import SimpleITK as sitk
import os
file = os.listdir('/output/images/automated-petct-lesion-segmentation')[0]
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/images/automated-petct-lesion-segmentation/lymphoma_ct.mha'))
mse = sum(sum(sum((output - expected_output) ** 2)))
if mse == 0.0:
    print('Test passed!')
else:
    print('Test failed!')
"""

docker volume rm priornet-output-$VOLUME_SUFFIX

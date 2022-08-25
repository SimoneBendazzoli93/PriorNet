# PriorNet
Cascade model ( Tumor appearance extraction + multi-channel nnUNet ) for AutoPET Challenge 2022

## Test PriorNet

To test PriorNet, run [test.sh](test.sh), specifying **TESTPATH**  and **TESTOUTPUTPATH** with the respective folder path for
the input and the ground truth to test.

## Export PriorNet

To export the PriorNet Docker image, run [export.sh](export.sh)
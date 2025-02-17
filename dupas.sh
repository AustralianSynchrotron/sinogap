#!/bin/bash


if [ -z "${1}" ] ; then
  echo "No input dir (first argument) was given." >&2
  exit 1
fi
iDir="${1}"

if [ -z "${2}" ] ; then
    echo "No id (cuda device) - second argument was given." >&2
  exit 1
fi
devNum="${2}"

rm -rf runs/experiment_$devNum/
cp -v -r ${iDir}/logs runs/experiment_$devNum
cp -v ${iDir}/checkPoint.pth checkPoint_${devNum}.pth 
cp -v ${iDir}/train.hdf train_${devNum}.hdf 
cp -v ${iDir}/sinogap.ipynb sinogap_${devNum}.ipynb
cp -v ${iDir}/model_gen.pt model_${devNum}_gen.pt




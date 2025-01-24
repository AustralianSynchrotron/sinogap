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
cp -r ${iDir}/logs runs/experiment_$devNum
cp ${iDir}/checkPoint.pth checkPoint_${devNum}.pth 
cp ${iDir}/train.hdf train_${devNum}.hdf 
cp ${iDir}/sinogap.ipynb sinogap_${devNum}.ipynb
cp ${iDir}/model_gen.pt model_${devNum}_gen.pt




#!/usr/bin/env python3

import numpy as np
import argparse
import h5py
from matplotlib.image import imread

parser = argparse.ArgumentParser(description='Closes gaps in the sinogram.')
parser.add_argument('vollume', type=str, nargs='*', default="",
                    help='HDF5 volume of CT projections.')
parser.add_argument('mask', type=str, nargs='*', default="",
                    help='Tiff image size of the projections with the pixels < 1 to be closed.')
parser.add_argument('output', type=str, nargs='*', default="",
                    help='Output HDF5 volume.')
args = parser.parse_args()


# Prepare HDF to read
def dataFromHDF(inputString):
    sampleHDF = inputString.split(':')
    if len(sampleHDF) != 2 :
        raise Exception(f"String \"{inputString}\" does not represent an HDF5 format \"fileName:container\".")
    try :
        trgH5F =  h5py.File(sampleHDF[0],'r')
    except :
        raise Exception(f"Failed to open HDF file '{sampleHDF[0]}'.")
    if  sampleHDF[1] not in trgH5F.keys():
        raise Exception(f"No dataset '{sampleHDF[1]}' in input file {sampleHDF[0]}.")
    data = trgH5F[sampleHDF[1]]
    if not data.size :
        raise Exception(f"Container \"{inputString}\" is zero size.")
    sh = data.shape
    if len(sh) != 3 :
        raise Exception(f"Dimensions of the container \"{inputString}\" is not 3: {sh}.")
    return data


data = dataFromHDF(args.volume)
faceSh = data.shape[1:]


def loadImage(imageName, expectedShape=None) :
    if not imageName:
        return None
    imdata = imread(imageName).astype(np.float32)
    if len(imdata.shape) != 2 :
        raise Exception(f"Image {imageName} is not grayscle.")
    if not expectedShape is None and expectedShape != imdata.shape :
        raise Exception(f"Shape of the image {imageName} is {imdata.shape} where {expectedShape} was expected.")
    return imdata

mask = loadImage(args.mask, faceSh)




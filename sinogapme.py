#!/usr/bin/env python3



import argparse


parser = argparse.ArgumentParser(description='Fill missing data using sinogap model.')
parser.add_argument('input', type=str, default="",
                    help='Input stack of CT projections as file.hdf@dataset or single sinogram as tif image.')
parser.add_argument('output', type=str, default="",
                    help='Output HDF5 file.')
parser.add_argument('-m', '--mask', type=str, default="",
                    help='Mask of the input stack for hdf volume. If not given, then zeros are filled.')
parser.add_argument('-z', '--zeros',  action='store_true', default=False,
                    help='Forcebly fill gaps as mask defines, even if the corresponding pixels in the input volume are non-zero.')
#parser.add_argument('-M', '--model', type=str, default="",
#                    help='Model to use.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Be verbose.')

args = parser.parse_args()

import os
from pathlib import Path
import h5py
import tifffile
import tqdm

import math
import numpy as np
import torch
from torchvision import transforms


myPath = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda:0')
maxBatchSize = 1
cfgPath = os.path.join(myPath, ".local.cfg")
if os.path.isfile(cfgPath):
    try:
        localCfgDict = dict()
        exec(open(cfgPath).read(),localCfgDict)
        if 'torchdevice' in localCfgDict :
            device = torch.device(localCfgDict['torchdevice'])
        if 'sinogap_maxBatchSize' in localCfgDict :
            maxBatchSize = localCfgDict['sinogap_maxBatchSize']
    except KeyError:
        raise
    except:
        pass


#%% I/O


hdfDelimiter = '@' # delimiter between filename and dataset path


def residesInMemory(hdfName) :
    mmapPrefixes = ["/dev/shm",]
    if "CTAS_MMAP_PATH" in os.environ :
        mmapPrefixes.extend(os.environ["CTAS_MMAP_PATH"].split(':'))
    hdfName = os.path.realpath(hdfName)
    for mmapPrefix in mmapPrefixes :
        if hdfName.startswith(mmapPrefix) :
            return True
    return False


def mmapMeIfYouCan(trgH5F, data, mode='r+') :
    if not residesInMemory(trgH5F.filename) :
        return None
    fileSize = trgH5F.id.get_filesize()
    offset = data.id.get_offset()
    dtype = data.id.dtype
    plist = data.id.get_create_plist()
    if offset is None \
    or offset < 0 \
    or not plist.get_layout() in (h5d.CONTIGUOUS, h5d.COMPACT) \
    or plist.get_external_count() \
    or plist.get_nfilters() \
    or fileSize - offset < math.prod(data.shape) * data.dtype.itemsize :
        return None
    # now all is ready
    name = trgH5F.filename
    trgH5F.close()
    dataN = np.memmap(name, shape=data.shape, dtype=dtype, mode=mode, offset=offset)
    data = dataN
    #plist = trgH5F.id.get_access_plist()
    #fileno = trgH5F.id.get_vfd_handle(plist)
    #dataM = mmap.mmap(fileno, fileSize, offset=offset, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    return data


def getInData(inputString, verbose=False, preread=False):
    """
    Function accesses a dataset from read-only HDF5 file and returns it.

    If file resides in memory (or in one of the the paths from the CTAS_MMAP_PATH environment variable)
    this function tries to mmap dataset into memory, build numpy array on top of it and returns it.
    If mmaping is not possible then given dataset is accessed and, if preread is False, is returned as is;
    with preread=True the dataset is read into into numpy array which is then returned.

    :param inputString: Filename of the HDF5 file and dataset path inside it;
                        Two components are separated by the delimiter (see below).
                        F.e. "filename.hdf@/data" (with @ as the delimiter).
    :param verbose: prints some messages.
    :param preread: only makes sense for files which cannot be mmapped. If True then dataset is read into
                    numpy array which is returned. If false h5py.Dataset is returned.

    :return: If file can be mmaped into memory or preread=True, returns numpy array with the dataset.
             Otherwise h5py.Dataset is returned.
    """

    global hdfDelimiter

    nameSplit = inputString.split(hdfDelimiter)
    if len(nameSplit) != 2 :
        raise Exception(f"String \"{inputString}\" does not represent an HDF5 format \"fileName{hdfDelimiter}container\".")
    hdfName = nameSplit[0]
    hdfVolume = nameSplit[1]
    try :
        trgH5F =  h5py.File(hdfName,'r', swmr=True)
    except :
        raise Exception(f"Failed to open HDF file '{hdfName}'.")
    if  hdfVolume not in trgH5F.keys():
        raise Exception(f"No dataset '{hdfVolume}' in input file {hdfName}.")
    data = trgH5F[hdfVolume]
    if not data.size :
        raise Exception(f"Container \"{inputString}\" is zero size.")
    sh = data.shape
    if len(sh) != 3 :
        raise Exception(f"Dimensions of the container \"{inputString}\" is not 3: {sh}.")
    mmaped = mmapMeIfYouCan(trgH5F, data, mode='r')
    if mmaped is None :
        if preread :
          dataN = np.empty(data.shape, dtype=np.float32)
          if verbose :
              print(f"Reading input \"{inputString}\" of {data.shape} size ... ", end="", flush=True)
          data.read_direct(dataN)
          if verbose :
              print("Done.")
          data = dataN
          trgH5F.close()
    else :
        data = mmaped
    return data


def getOutData(outputString, shape=None, dtype=None, overwrite=False) :

    global hdfDelimiter

    if shape is not None :
        if len(shape) == 2 :
            shape = (1,*shape)
        if len(shape) != 3 :
            raise Exception(f"Not appropriate output array size {shape}.")

    nameSplit = outputString.split(hdfDelimiter)
    if len(nameSplit) != 2 :
        raise Exception(f"String \"{outputString}\" does not represent an HDF5 format \"fileName{hdfDelimiter}container\".")
    hdfName = nameSplit[0]
    hdfVolume = nameSplit[1]
    try :
        trgH5F =  h5py.File(hdfName,'a', libver='latest')
    except :
        raise Exception(f"Failed to open HDF file '{hdfName}'.")

    data = None
    if hdfVolume in trgH5F :
        data = trgH5F[hdfVolume]
        if not overwrite :
            if shape is not None and data.shape != shape :
                raise Exception(f"Shape of dataset \"{outputString}\" {data.shape} is not equal to requested {shape}.")
            if dtype is not None and data.dtype != dtype :
                raise Exception(f"Data type of dataset \"{outputString}\" {data.dtype} is not equal to requested {dtype}.")
        elif ( shape is not None and data.shape != shape ) or ( dtype is not None and data.dtype != dtype ) :
            del trgH5F[hdfVolume]
            data = None
    if data is None :
        if shape is None :
            raise Exception(f"No dataset \"{outputString}\" exists and no shape was provided to create it.")
        if dtype is None :
            raise Exception(f"No dataset \"{outputString}\" exists and no data type was provided to create it.")
        data = trgH5F.create_dataset(hdfVolume, shape=shape, dtype=dtype)
        # TODO : check other possible preallocations
        data[-1,-1,-1]=0
    mmaped = mmapMeIfYouCan(trgH5F, data, mode='r+')
    if mmaped is not None :
        data = mmaped
    return data


def closeOutData(data) :
    if isinstance(data, isinstance(data, np.memmap)) :
        data.flush()
        data._mmap.close()
    elif isinstance(data, h5py.Dataset) :
        data.file.close()




#%% MODELS

import sinogap_model as sinogap

model = sinogap.model(16, 'gen', False, 16, 16, "sinogap_model.pt")
_=model.requires_grad_(False)
_=model.eval()

#%% EXEC



def fillSinogram(sinogram, mask=None, fill_zerosOnly=True) :
    global device

    sinogram = torch.tensor(sinogram).to(device=device)
    if mask is None :
        mask =  torch.where( torch.amin( sinogram, dim=-2) == 0 , 0, 1 )
    sinoLen = sinogram.shape[-1]

    def fillStripe(stripe) :

        gapW = model.cfg.gapW
        ssh = model.cfg.sinoSh

        stripe = stripe.to(device)
        stripe, _ = sinogap.unsqeeze4dim(stripe)
        sinoW = stripe.shape[-1]
        sinoL = stripe.shape[-2]
        if sinoW % 7 :
            raise Exception(f"Sinogram width {sinoW} is not devisable by 7.")
        blockW = sinoW // 7
        resizedSino = torch.zeros(( 1 , 1 , sinoL , ssh[1] ), device=device)
        resizedSino[ ... , : 3*gapW ] = torch.nn.functional.interpolate(
            stripe[ ... , : 3*blockW ], size=( sinoL , 3*gapW ), mode='bilinear')
        resizedSino[ ... , 3*gapW : 4*gapW ] = torch.nn.functional.interpolate(
            stripe[ ... , 3*blockW : 4*blockW ], size=( sinoL , gapW ), mode='bilinear')
        resizedSino[ ... , 4*gapW:] = torch.nn.functional.interpolate(
            stripe[ ... , 4*blockW : ], size=( sinoL , 3*gapW ), mode='bilinear')

        resizedSino = torch.nn.functional.interpolate(resizedSino, size=ssh, mode='bilinear')
        resizedSino = model.forward(resizedSino)
        resizedSino = torch.nn.functional.interpolate(resizedSino, size=stripe.shape[-2:], mode='bilinear')

        if fill_zerosOnly :
            stripe[ ... , 3*blockW : 4*blockW ] =  torch.where( stripe   [ ... , 3*blockW : 4*blockW ] == 0,
                                                                  resizedSino[ ... , 3*blockW : 4*blockW ],
                                                                  stripe   [ ... , 3*blockW : 4*blockW ] )
        else :
            stripe[ ... , 3*blockW : 4*blockW ] = resizedSino[ ... , 3*blockW : 4*blockW ]

        return stripe


    def closeGapsFromList(gapsIn, ignoreMargins = False) :
        gapsToRet = []
        for gapI, gap in enumerate(gapsIn) :
            gapW = gap.stop - gap.start
            sideW = 3*gapW
            prevGap = gapsToRet[-1].stop if len(gapsToRet) else 0
            nextGap = gapsIn[gapI+1].start if gapI < len(gapsIn)-1 else sinoLen
            if  gapW <= 32 \
            and ( ignoreMargins or gap.start - prevGap > sideW ) \
            and ( ignoreMargins or nextGap - gap.stop > sideW ) :
                stripeRange=np.s_[ max(0,gap.start - sideW) : min(sinoLen, gap.stop + sideW) ]
                stripeData = sinogram[:,stripeRange]
                filledData = fillStripe(stripeData).squeeze()
                sinogram[:,stripeRange] = filledData
            else :
                #print(f"Warning. Gap {gap} does not have enough space"
                #      f" between adjacent gaps {np.s_[prevGap,nextGap]} to process. "
                #      f" will try in the next iteration.")
                gapsToRet.append(gap)
        return gapsToRet


    gaps = []
    clmn=0
    gapStart=-1
    while clmn < sinoLen :
        value = mask[clmn]
        if ( value < 1 and gapStart >= 0 ) or \
           ( value >= 1 and gapStart < 0 ) :
               clmn +=1
               continue
        if value < 1 :
            if gapStart < 0  : # start the gap
                gapStart = clmn
        else :
            if gapStart >= 0  : # end the gap
                gaps.append(np.s_[gapStart:clmn])
                gapStart = -1
        clmn += 1
    if gapStart >= 0:
        gaps.append(np.s_[gapStart:sinoLen])

    # trivial gaps
    while True :
        gapsOnEnter = len(gaps)
        gaps = closeGapsFromList(gaps)
        if not len(gaps) or len(gaps) == gapsOnEnter :
            break
    curGap = 0
    # double gaps with total width less than 32
    while curGap < len(gaps)-1 : # try to combine gaps
        combGaps = [ np.s_[ gaps[curGap].start : gaps[curGap+1].stop ], ]
        gapLeft = len(closeGapsFromList(combGaps))
        if gapLeft :
            curGap += 1
        else :
            gaps = [ *gaps[:curGap], *gaps[curGap+2:] ]
    # rest of the gaps, ignore margins
    _ = closeGapsFromList(gaps, ignoreMargins=True)
    _ = closeGapsFromList(gaps, ignoreMargins=True)

    return sinogram





try :   # single tiff input
    sinogram = tifffile.imread(args.input).astype(np.float32)
    sinogram = fillSinogram(sinogram).cpu().numpy()
    tifffile.imwrite(args.output, sinogram)
    exit(0)
except Exception as e :
    #print("Exception: ",e)
    pass


if args.verbose :
    print("Reading input ...", end="", flush=True)
inData = getInData(args.input, preread=False)
fsh = inData.shape[1:]
if len(args.mask) :
    mask = tifffile.imread(args.mask).astype(np.float32) if len(args.mask) else None
    if mask.shape != fsh :
        raise Exception(f"Different image and mask shapes: {fsh} and {mask.shape}.")
else :
    mask = None
leftMask = np.ones(fsh, dtype=np.uint8)
outData = getOutData(args.output, inData.shape, inData.dtype)
if args.verbose :
    print(" Read.")
    print(" Filling.")

pbar = tqdm.tqdm(total=fsh[-2]) if args.verbose else None
for curSl in range(fsh[-2]):
    inSinogram = inData[:,curSl,:]
    inMask = None if mask is None else mask[curSl,:]
    outSinogram = fillSinogram(inSinogram, inMask, args.zeros)
    outData[:,curSl,:] = outSinogram.cpu().numpy()
    if pbar is not None:
        pbar.update(1)

if pbar is not None:
    pbar.close()
    print("Done")

if not np.all(leftMask > 0) :
    leftName = ".".join(args.output.split(".")[:-1]) + "_left.tif"
    if args.verbose :
        print(f"Some pixels left cannot be filled. Saving their mask into '{leftName}'")
    tifffile.imwrite( leftName, leftMask * 255 )


#leftMask4fill = leftMask.copy()
#leftMask4stitch = leftMask.copy()
#for row in range(fsh[0]) :
#    if np.all(leftMask[row,:]==0) :
#        leftMask4fill[row,:] = 1
#        leftMask4stitch[row,:] = 0
#    else :
#        leftMask4fill[row,:] = leftMask[row,:]
#        leftMask4stitch[row,:] = 1
#leftMask4fill *= 255
#leftMask4stitch *= 255
##leftMask *= 255
#leftMaskName = ".".join(args.output.split(".")[:-1])+"_mask"
#tifffile.imwrite(leftMaskName + ".tif", leftMask4stitch)
#if not np.all(leftMask4fill) :
#    tifffile.imwrite(leftMaskName + "_left.tif", leftMask4fill)




# %%

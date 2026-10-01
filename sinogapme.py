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

import math
import numpy as np
from sympy import false
import torch
from torchvision import transforms

import tifffile
import tqdm

import sinogap_module as sg

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
sg.device = device



#%% MODELS

# easier to keep than to clean the code
sg.TCfg = sg.TCfgClass(
     exec = 3
    ,nofEpochs = None
    ,latentDim = 64
    ,batchSize = 1
    ,batchSplit = 1
    ,labelSmoothFac = 0.1
    ,learningRateD = 1
    ,learningRateG = 1
    ,dataDir=""
)

sg.DCfg = sg.DCfgClass(16,False)
sg.brickMasks = sg.createBrickMasks()


layers2 = [
    ( 1, 2, 3, 2, 1),
    ( 2, 2, 3, (2,1), (1,0)),
    ( 4, 2, 3, (2,1), (1,0)),
]
class Generator2(sg.GeneratorTemplate) :
    def __init__(self):
        super().__init__(2, 1, 16, 16, layers2)
        self.lowResGenerator = None # Generator4()


layers4 = [
    ( 1, 2, 3, 2, 1),
    ( 2, 2, 3, 2, 1),
    ( 4, 2, 3, (2,1), (1,0)),
    ( 8, 2, 3, (2,1), (1,0)),
]
class Generator4(sg.GeneratorTemplate) :
    def __init__(self):
        super().__init__(4, 2, 16, 16, layers4)
        self.lowResGenerator = Generator2()



layers8 = [
    ( 1, 2, 3, 2, 1),
    ( 2, 2, 3, 2, 1),
    ( 4, 2, 3, 2, 1),
    ( 8, 2, 3, (2,1), (1,0)),
    (16, 2, 3, (2,1), (1,0)),
]
class Generator8(sg.GeneratorTemplate) :
    def __init__(self):
        super().__init__(8, 2, 16, 16, layers8)
        self.lowResGenerator = Generator4()


layers16 = [
    ( 1, 2, 3, 2, 1),
    ( 2, 2, 3, 2, 1),
    ( 4, 2, 3, 2, 1),
    ( 8, 2, 3, 2, 1),
    (16, 2, 3, (2,1), (1,0)),
    (32, 2, 3, (2,1), (1,0)),
]
class Generator16(sg.GeneratorTemplate) :
    def __init__(self):
        super().__init__(16, 2, 16, 16, layers16)
        self.lowResGenerator = Generator8()


sg.generator = Generator16().to(device)
_=sg.load_model(sg.generator, os.path.join(myPath, "sinogap_model.pt") )
_=sg.generator.requires_grad_(False)
_=sg.generator.eval()

model = sg.generator

#%% EXEC



def fillSinogram(sinogram, mask=None, fill_zerosOnly=True) :

    sinogram = torch.tensor(sinogram).to(device=model.device())
    if mask is None :
        mask =  torch.where( torch.amin( sinogram, dim=-2) == 0 , 0, 1 )
    sinoLen = sinogram.shape[-1]

    def fillStripe(stripe) :

        gapW = model.cfg.gapW
        ssh = model.cfg.sinoSh

        stripe = stripe.to(device)
        stripe, _ = sg.unsqeeze4dim(stripe)
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
inData = sg.getInData(args.input, preread=False)
fsh = inData.shape[1:]
mask = sg.loadImage(args.mask, fsh) if len(args.mask) else None
leftMask = np.ones(fsh, dtype=np.uint8)
outData = sg.getOutData(args.output, inData.shape, inData.dtype)
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

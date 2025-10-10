
from re import sub
from weakref import ref
import IPython

import sys
import os
import random
import time
import gc
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import glob

import math
import statistics
from cv2 import norm
import numpy as np
import test
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torchvision
from torch import optim, rand, randint
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import h5py
from h5py import h5d
import tifffile
import tqdm

import ssim


def initIfNew(var, val=None) :
    if var in locals() :
        return locals()[var]
    if var in globals() :
        return globals()[var]
    return val

@dataclass
class TCfgClass:
    exec : int
    latentDim: int
    batchSize: int
    labelSmoothFac: float
    learningRateD: float
    learningRateG: float
    dataDir : str
    device: torch.device = torch.device('cpu')
    batchSplit : int = 1 # negative to load multiple batches at a time.
    nofEpochs: int = 0
    num_workers : int = os.cpu_count()
    historyHDF : str = field(repr = True, init = False)
    logDir : str = field(repr = True, init = False)
    def __post_init__(self):
        if self.device == torch.device('cpu')  :
            self.device = torch.device(f"cuda:{self.exec}")
        self.historyHDF = f"train_{self.exec}.hdf"
        self.logDir = f"runs/experiment_{self.exec}"
        if self.batchSplit > 1 and self.batchSize % self.batchSplit :
            raise Exception(f"Batch size {self.batchSize} is not divisible by batch split {self.batchSplit}.")
global TCfg
TCfg = initIfNew('TCfg')


@dataclass
class DCfgClass:
    gapW : int
    brick : bool = field(repr = False)
    sinoSh : tuple = field(repr = True, init = False)
    gapSh : tuple = field(repr = True, init = False)
    gapRngX : type(np.s_[:]) = field(repr = True, init = False)
    gapRng : type(np.s_[:]) = field(repr = True, init = False)
    readSh : tuple = field(repr = True, init = False)
    def __post_init__(self):
        self.readSh : tuple = (128 if self.brick else None ,128)
        self.sinoSh = ( (8 if self.brick else 256) * self.gapW , 8*self.gapW )
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[...,self.gapRngX]
DCfg = initIfNew('DCfg')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def plotData(dataY, rangeY=None, dataYR=None, rangeYR=None,
             dataX=None, rangeX=None, rangeP=None,
             figsize=(16,8), saveTo=None, show=True):

    if type(dataY) is np.ndarray :
        plotData((dataY,), rangeY=rangeY, dataYR=dataYR, rangeYR=rangeYR,
             dataX=dataX, rangeX=rangeX, rangeP=rangeP,
             figsize=figsize, saveTo=saveTo, show=show)
        return
    if type(dataYR) is np.ndarray :
        plotData(dataY, rangeY=rangeY, dataYR=(dataYR,), rangeYR=rangeYR,
             dataX=dataX, rangeX=rangeX, rangeP=rangeP,
             figsize=figsize, saveTo=saveTo, show=show)
        return
    if type(dataY) is not tuple :
        eprint(f"Unknown data type to plot: {type(dataY)}.")
        return
    if type(dataYR) is not tuple and dataYR is not None:
        eprint(f"Unknown data type to plot: {type(dataYR)}.")
        return

    last = min( len(data) for data in dataY )
    if dataYR is not None:
        last = min( last,  min( len(data) for data in dataYR ) )
    if dataX is not None:
        last = min(last, len(dataX))
    if rangeP is None :
        rangeP = (0,last)
    elif type(rangeP) is int :
        rangeP = (0,rangeP) if rangeP > 0 else (-rangeP,last)
    elif type(rangeP) is tuple :
        rangeP = ( 0    if rangeP[0] is None else rangeP[0],
                   last if rangeP[1] is None else rangeP[1],)
    else :
        eprint(f"Bad data type on plotData input rangeP: {type(rangeP)}")
        raise Exception(f"Bug in the code.")
    rangeP = np.s_[ max(0, rangeP[0]) : min(last, rangeP[1]) ]
    if dataX is None :
        dataX = np.arange(rangeP.start, rangeP.stop)

    #plt.style.use('default')
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.xaxis.grid(True, 'both', linestyle='dotted')
    if rangeX is not None :
        ax1.set_xlim(rangeX)
    else :
        ax1.set_xlim(rangeP.start,rangeP.stop-1)

    ax1.yaxis.grid(True, 'both', linestyle='dotted')
    nofPlots = len(dataY)
    if rangeY is not None:
        ax1.set_ylim(rangeY)
    colors = [ matplotlib.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
    for idx , data in enumerate(dataY):
        ax1.plot(dataX, data[rangeP], linestyle='-',  color=colors[idx])

    if dataYR is not None : # right Y axis
        ax2 = ax1.twinx()
        ax2.yaxis.grid(True, 'both', linestyle='dotted')
        nofPlots = len(dataYR)
        if rangeYR is not None:
            ax2.set_ylim(rangeYR)
        colors = [ matplotlib.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
        for idx , data in enumerate(dataYR):
            ax2.plot(dataX, data[rangeP], linestyle='dashed',  color=colors[idx])

    if saveTo:
        fig.savefig(saveTo)
    if not show:
        plt.close(fig)


def plotImage(image, frameon=False) :
    plt.figure(frameon=frameon)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()


def plotImages(images, frameon=False) :
    plt.figure(frameon=frameon)
    for i, img in enumerate(images) :
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis("off")
    plt.show()


def sliceShape(shape, sl) :
    if type(shape) is int :
        shape = torch.Size([shape])
    if type(sl) is tuple :
        if len(shape) != len(sl) :
            raise Exception(f"Different sizes of shape {shape} and sl {sl}")
        out = []
        for i in range(0, len(shape)) :
            indeces = sl[i].indices(shape[i])
            out.append(indeces[1]-indeces[0])
        return out
    elif type(sl) is slice :
        indeces = sl.indices(shape[0])
        return indeces[1]-indeces[0]
    else :
        raise Exception(f"Incompatible object {sl}")


def tensorStat(stat) :
    print(f"{stat.mean().item():.3e}, {stat.std().item():.3e}, "
          f"{stat.min().item():.3e}, {stat.max().item():.3e}")


def fillWheights(seq, std=0.001) :
    for wh in seq :
        if hasattr(wh, 'weight') :
            #torch.nn.init.xavier_uniform_(wh.weight)
            #torch.nn.init.zeros_(wh.weight)
            #torch.nn.init.constant_(wh.weight, 0)
            #torch.nn.init.uniform_(wh.weight, a=0.0, b=1.0, generator=None)
            torch.nn.init.normal_(wh.weight, mean=0.0, std=std)
        if hasattr(wh, 'bias') and wh.bias is not None :
            torch.nn.init.normal_(wh.bias, mean=0.0, std=0)


def unsqeeze4dim(tens):
    orgDims = tens.dim()
    if tens.dim() == 2 :
        tens = tens.unsqueeze(0)
    if tens.dim() == 3 :
        tens = tens.unsqueeze(1)
    return tens, orgDims


def squeezeOrg(tens, orgDims):
    if orgDims == tens.dim():
        pass
    if tens.dim() != 4 or orgDims > 4 or orgDims < 2:
        raise Exception(f"Unexpected dimensions to squeeze: {tens.dim()} {orgDims}.")
    if orgDims < 4 :
        if tens.shape[1] > 1:
            raise Exception(f"Cant squeeze dimension 1 in: {tens.shape}.")
        tens = tens.squeeze(1)
    if orgDims < 3 :
        if tens.shape[0] > 1:
            raise Exception(f"Cant squeeze dimension 0 in: {tens.shape}.")
        tens = tens.squeeze(0)
    return tens


def set_seed(SEED_VALUE):
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    np.random.seed(SEED_VALUE)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    return


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=TCfg.device))
    return model



def addToHDF(filename, containername, data) :
    if len(data.shape) == 2 :
        data=np.expand_dims(data, 0)
    if len(data.shape) != 3 :
        raise Exception(f"Not appropriate input array size {data.shape}.")

    with h5py.File(filename,'a') as file :

        if  containername not in file.keys():
            dset = file.create_dataset(containername, data.shape,
                                       maxshape=(None,data.shape[1],data.shape[2]),
                                       dtype='f')
            dset[()] = data
            return

        dset = file[containername]
        csh = dset.shape
        if csh[1] != data.shape[1] or csh[2] != data.shape[2] :
            raise Exception(f"Shape mismatch: input {data.shape}, file {dset.shape}.")
        msh = dset.maxshape
        newLen = csh[0] + data.shape[0]
        if msh[0] is None or msh[0] >= newLen :
            dset.resize(newLen, axis=0)
        else :
            raise Exception(f"Insufficient maximum shape {msh} to add data"
                            f" {data.shape} to current volume {dset.shape}.")
        dset[csh[0]:newLen,...] = data
        file.close()


    return 0



def loadImage(imageName, expectedShape=None) :
    if not imageName:
        return None
    #imdata = imread(imageName).astype(np.float32)
    imdata = tifffile.imread(imageName).astype(np.float32)
    if len(imdata.shape) == 3 :
        imdata = np.mean(imdata[:,:,0:3], 2)
    if not expectedShape is None  and  imdata.shape != expectedShape :
        raise Exception(f"Dimensions of the input image \"{imageName}\" {imdata.shape} "
                        f"do not match expected shape {expectedShape}.")
    return imdata



def residesInMemory(hdfName) :
    mmapPrefixes = ["/dev/shm",]
    if "CTAS_MMAP_PATH" in os.environ :
        mmapPrefixes.extend(os.environ["CTAS_MMAP_PATH"].split(':'))
    hdfName = os.path.realpath(hdfName)
    for mmapPrefix in mmapPrefixes :
        if hdfName.startswith(mmapPrefix) :
            return True
    return False

def goodForMmap(trgH5F, data) :
    fileSize = trgH5F.id.get_filesize()
    offset = data.id.get_offset()
    plist = data.id.get_create_plist()
    if offset < 0 \
    or not plist.get_layout() in (h5d.CONTIGUOUS, h5d.COMPACT) \
    or plist.get_external_count() \
    or plist.get_nfilters() \
    or fileSize - offset < math.prod(data.shape) * data.dtype.itemsize :
        return None, None
    else :
        return offset, data.id.dtype


def getInData(inputString, verbose=False, preread=False):
    nameSplit = inputString.split(':')
    if len(nameSplit) == 1 : # tiff image
        data = loadImage(nameSplit[0])
        data = np.expand_dims(data, 1)
        return data
    if len(nameSplit) != 2 :
        raise Exception(f"String \"{inputString}\" does not represent an HDF5 format \"fileName:container\".")
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
    try : # try to mmap hdf5 if it is in memory
        if not residesInMemory(hdfName) :
            raise Exception()
        fileSize = trgH5F.id.get_filesize()
        offset = data.id.get_offset()
        dtype = data.id.dtype
        plist = data.id.get_create_plist()
        if offset < 0 \
        or not plist.get_layout() in (h5d.CONTIGUOUS, h5d.COMPACT) \
        or plist.get_external_count() \
        or plist.get_nfilters() \
        or fileSize - offset < math.prod(sh) * data.dtype.itemsize :
            raise Exception()
        # now all is ready
        dataN = np.memmap(hdfName, shape=sh, dtype=dtype, mode='r', offset=offset)
        data = dataN
        trgH5F.close()
        #plist = trgH5F.id.get_access_plist()
        #fileno = trgH5F.id.get_vfd_handle(plist)
        #dataM = mmap.mmap(fileno, fileSize, offset=offset, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    except :
        if preread :
            dataN = np.empty(data.shape, dtype=np.float32)
            if verbose :
                print("Reading input ... ", end="", flush=True)
            data.read_direct(dataN)
            if verbose :
                print("Done.")
            data = dataN
            trgH5F.close()
    return data



def createWriter(logDir, addToExisting=False) :
    if not addToExisting and os.path.exists(logDir) :
        raise Exception(f"Log directory \"{logDir}\" for the experiment already exists."
                        " Remove it or implicitry overwrite with setting addToExisting to True.")
    return SummaryWriter(logDir)
writer = initIfNew('writer')


class DevicePlace:
    def __call__(self, x):
        return x.to(TCfg.device)



class StripesFromHDF :

    # Setting exclusize to True makes dataset consisting only out of non-overlapping sinograms
    def __init__(self, sampleName, maskName, exclusive=False):
        self.volume = getInData(sampleName, False, False)
        self.sh = self.volume.shape
        self.fsh = self.sh[1:3]
        self.mask = loadImage(maskName, self.fsh)
        self.mask /= self.mask.max()
        if self.mask is None :
            self.mask = np.ones(self.fsh, dtype=np.uint8)
        #self.mask = self.mask.astype(bool)
        forbidenSinos = self.mask.copy()
        for shft in range (1, DCfg.readSh[-1]) :
            forbidenSinos[:,:-shft] *= self.mask[:,shft:]
        forbidenSinos[:, -DCfg.readSh[-1]:] = 0
        if exclusive : # non-overlapping sinograms
            for yCr in range(0,self.fsh[0]) :
                xCr = 0
                while xCr < self.fsh[1]-DCfg.readSh[-1] :
                    if np.all(forbidenSinos[yCr, xCr:xCr+DCfg.readSh[-1]] > 0) :
                        forbidenSinos[ yCr, xCr+1 : xCr+DCfg.readSh[-1] ] = 0
                        xCr += DCfg.readSh[-1]
                    else :
                        forbidenSinos[ yCr, xCr ] = 0
                        xCr += 1
        self.allAvailableSinos = np.argwhere(forbidenSinos)
        self.availableFragments = 1 if DCfg.readSh[0] is None else \
            ( (self.sh[0] - DCfg.readSh[0] + 1) // ( DCfg.readSh[0] if exclusive else 1 ) )


    def __len__(self):
        return self.allAvailableSinos.shape[0] * self.availableFragments


    def __getitem__(self, index=None):
        if index is None :
            index = random.randint(0, len(self)-1)
            return self.__getitem__(index)
        elif isinstance(index, int) :
            fdx, zdx = divmod(index, self.availableFragments)
            #ydx, xdx = tuple(self.availableSinos[fdx])
            ydx, xdx = tuple( int(dx) for dx in self.allAvailableSinos[fdx,:] )
            #ydx, xdx = tuple( int(dx) for dx in self.exposedSinos[fdx,:] )
            return self.__getitem__((zdx, ydx, xdx))
        elif isinstance(index, tuple)  and len(index) == 3 :
            zdx = 0 if DCfg.readSh[0] is None else index[0]
            data = self.volume[ zdx : -1 if DCfg.readSh[0] is None else (zdx+DCfg.readSh[0]),
                                index[1],
                                index[2]:index[2]+DCfg.readSh[1]
                              ].copy()
            return data, index
        raise Exception(f"Bad index for data set: {index}.")


class StripesFromHDFs :

    def __init__(self, bases, exclusive=False):
        self.collection = []
        for base in bases :
            print(f"Loading train set {len(self.collection)+1} of {len(bases)}: " + base + " ... ", end="")
            self.collection.append(
                StripesFromHDF(f"{base}.hdf:/data", f"{base}.mask++.tif", exclusive) )
            print("Done")

    def __getitem__(self, index=None):
        if index is None:
            index = random.randint(0,len(self)-1)
            return self.__getitem__(index)
        elif isinstance(index, int) :
            leftover = index
            for setdx in range(len(self.collection)) :
                setLen = len(self.collection[setdx])
                if leftover >= setLen :
                    leftover -= setLen
                else :
                    data, subIndex = self.collection[setdx].__getitem__(leftover)
                    return data, (setdx, *subIndex)
        elif type(index) is tuple and len(index) == 4 :
            return self.collection[index[0]].__getitem__(index[1:])[0], index
        raise Exception(f"Bad index for collection of data sets: {index}.")

    def __len__(self):
        return sum( [ len(set) for set in self.collection ] )


    def get_dataset(self, transform=None, expose=1, shuffle=False) :

        class Sinos(torch.utils.data.Dataset) :
            def __init__(self, root, transform=None, expose=1, shuffle=False):
                self.container = root
                if not ( 0 < expose <= 1 ) :
                    raise f"Provided exposure {expose} is outside (0,1] range."
                self.expose = expose
                self.shuffle = shuffle
                self.transform = transform
                self.oblTransform = transforms.Compose( [transforms.ToTensor(),
                                                         #DevicePlace(),
                                                         transforms.Resize(DCfg.sinoSh)] )
            def __len__(self):
                return int(self.container.__len__() * self.expose)
            def __getitem__(self, index=None, doTransform=True):
                if self.shuffle and isinstance(index, int) : # randomize dataset
                    index = random.randint(0,self.container.__len__()-1)
                data, rIndex = self.container.__getitem__(index)
                data = self.oblTransform(data)
                if doTransform and self.transform :
                    data = self.transform(data)
                return data, rIndex
            def originalSinoLen(self,setdx) :
                return self.container.collection[setdx].sh[0]

        return Sinos(self, transform, expose, shuffle)


listOfTrainData = [
    "18692a.ExpChicken6mGyShift",
    "23574.8965435L.Eiger.32kev_sft",
    "19022g.11-EggLard",
    "18692b.MinceO",
    "23574.8965435L.Eiger.32kev_org",
    "19736b.09_Feb.4176862R_Eig_Threshold-4keV",
    "20982b.04_774784R",
    "18515.Lamb1_Eiger_7m_45keV_360Scan",
    "19736c.8733147R_Eig_Threshold-8keV.SAMPLE_Y1",
    "18692b_input_PhantomM",
    "21836b.2024-08-15-mastectomies.4201381L.35kev.20Hz",
    "23574h.9230799R.35kev",
    "18515.Lamb4_Excised_Eiger_7m_30keV_360Scan.Y1",
    "18648.B_Edist.80keV_0m_Eig_Neoprene.Y2",
    "19932.10_8093920_35keV",
    "19932.14_2442231_23keV",
    "19932.16_4193759_60keV",
]
listOfTestData = [
    "19603a.Exposures.70keV_7m_Calf2_Threshold35keV_25ms_Take2",
    "22280a_input_Day_4_40keV_7m_Threshold20keV_50ms_Y04_no_shell__0.05deg",
    "18515.Lamb4_Eiger_5m_50keV_360Scan.SAMPLE_Y1",
    "18692b_input_Phantom0",
    #"19603a.ROI-CTs.50keV_7m_Eiger_Sheep1",
]

def createDataSet(path, listOfData, exclusive=False, expose=1) :
    #listOfData = [file.removesuffix(".hdf") for file in glob.glob(path+ "/*.hdf", recursive=False)]
    listOfData = [ '/'.join((path,file))  for file in listOfData ]
    print(listOfData)
    sinoRoot = StripesFromHDFs(listOfData, exclusive)
    transList = []
    #transList.append(transforms.Resize(DCfg.sinoSh))
    if not exclusive :
        transList.append(transforms.RandomHorizontalFlip()),
        transList.append(transforms.RandomVerticalFlip()),
    #transList.append(transforms.Normalize(mean=(0.5), std=(1)))
    mytransforms = transforms.Compose(transList)
    return sinoRoot.get_dataset( transform=mytransforms, expose=expose, shuffle = not exclusive )
trainSet = initIfNew('trainSet')
testSet = initIfNew('testSet')


def createDataLoader(tSet, num_workers=os.cpu_count()) :
    return torch.utils.data.DataLoader(
        dataset=tSet,
        batch_size = TCfg.batchSize * max(1, -TCfg.batchSplit) ,
        shuffle=False, # randomize dataset instead of the dataloader because it takes enormous amount of time otherwise
        num_workers=num_workers,
        drop_last=True
    )


examples = [
    #(11142, 3150), # (0, 417, 1877)
    (38576, 2560), # (3, 476, 2855)
    (26289, 6300), # (2, 280, 828)
    (24299, 7160), # (2, 113, 988)
    (3186, 2455), # (0, 119, 240)
    ]

def createReferences(tSet, majorIdx = 0) :
    if majorIdx :
        examples.insert(0, examples.pop(majorIdx))
    mytransforms = transforms.Compose([
            transforms.Resize(DCfg.sinoSh),
            #transforms.Normalize(mean=(0.5), std=(1))
    ])
    refImages = torch.empty((len(examples), 1, *DCfg.sinoSh), dtype=torch.float32).to(TCfg.device)
    refBoxes = []
    for idx, ex in enumerate(examples) :
        if DCfg.readSh[0] is None :
            index = (ex[0][0], 0, ex[0][1], ex[0][2])
            refBoxes.append( int(ex[1] * refImages.shape[-2]) )
        else :
            index = (ex[0][0], int(ex[1]*tSet.originalSinoLen(ex[0][0])) , ex[0][1], ex[0][2])
            refBoxes.append(0)
        data = tSet.__getitem__(index, doTransform=False)[0]
        refImages[idx,0,...] = mytransforms(data)


    refNoises = torch.randn((refImages.shape[0],TCfg.latentDim)).to(TCfg.device)
    return refImages, refNoises, refBoxes
refImages = initIfNew('refImages')
refNoises = initIfNew('refNoises')
refBoxes = initIfNew('refBoxes')


def showMe(tSet, index=None) :
    global refImages, refNoises
    index = random.randint(0,len(tSet)-1) if index is None else index
    image, rindex = tSet[index]
    image = image.squeeze().transpose(0,1)
    print(index, rindex)
    tensorStat(image)
    plotImage(image.cpu())
    return rindex

def normalizeImages(images) :
    images, orgDims = unsqeeze4dim(images)
    images = images.clone().detach()
    stds, means = torch.std_mean(images, dim=(-1,-2), keepdim=True)
    stds += 1e-7
    images = (images - means) / stds # normalize per image
    return images, (orgDims, stds, means)

def reNormalizeImages(images, norms) :
    images = images * norms[1][:,[0],...] + norms[2][:,[0],...] # renormalise
    images = squeezeOrg(images, norms[0])
    return images






class SubGeneratorTemplate(nn.Module):

    def __init__(self, gapW, brick, batchNorm=True, inChannels=1):
        super(SubGeneratorTemplate, self).__init__()
        self.cfg = DCfgClass(gapW, brick)
        self.lowResGenerator = None
        self.baseChannels = None
        self.inChannels = inChannels
        self.amplitude = 4
        self.batchNorm = batchNorm

    def encblock(self, chIn, chOut, kernel, stride=1, norm=None, padding=1) :
        if norm is None :
            norm = self.batchNorm
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias = not norm,
                                padding=padding, padding_mode='reflect')  )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)

    def decblock(self, chIn, chOut, kernel, stride=1, norm=None, padding=1, outputPadding=None) :
        if norm is None :
            norm = self.batchNorm
        if outputPadding is None :
            if isinstance(stride, int) :
                outputPadding = stride - 1
            else :
                outputPadding = tuple( strd - 1 for strd in stride )
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.ConvTranspose2d(chIn, chOut, kernel, stride=stride, bias = not norm,
                                          padding=padding, padding_mode='zeros', output_padding=outputPadding) )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)

    def createFClink(self) :
        smpl = torch.zeros((1, self.inChannels, *self.cfg.sinoSh))
        for encoder in self.encoders :
            smpl = encoder(smpl)
        encSh = smpl.shape
        linChannels = math.prod(encSh)
        toRet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        fillWheights(toRet)
        return toRet

    def createLastTouch(self, chIn=1) :
        toRet = nn.Sequential(
            nn.Conv2d(chIn*self.baseChannels+self.inChannels, 1, 1),
            nn.Tanh(),
        )
        fillWheights(toRet)
        return toRet


    def generateImages(self, images, noises=None) :
        res = images.clone()
        res[self.cfg.gapRng] = self.forward(images)[self.cfg.gapRng]
        return res


    def preFill(self, images) :
        images, orgDims = unsqeeze4dim(images)
        if self.cfg.gapW == 2:
            res = torch.zeros( (images.shape[0], 1 , *self.cfg.gapSh ), device=images.device)
            res[...,0] += 2*images[:,[0],:,self.cfg.gapRngX.start-1] + images[:,[0],:,self.cfg.gapRngX.stop]
            res[...,1] += 2*images[:,[0],:,self.cfg.gapRngX.stop] + images[:,[0],:,self.cfg.gapRngX.start-1]
            res /= 3
        elif self.lowResGenerator is None :
            res = images[:,[0],*self.cfg.gapRng]
        else :
            preImages = torch.nn.functional.interpolate(images, scale_factor=0.5, mode='area')
            #with torch.set_grad_enabled(not self.lowResGenerator is None) :
            res = self.lowResGenerator.forward(preImages)[self.lowResGenerator.cfg.gapRng]
            res = torch.nn.functional.interpolate(res, scale_factor=2, mode='bilinear')
        return squeezeOrg(res, orgDims)

    def dropIN(self,images) :
        dwTrain = [images,]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
            #print(f"{dwTrain[-2].shape} -> {dwTrain[-1].shape}")
        #return dwTrain[-1]
        mid = self.fcLink(dwTrain[-1])
        #return mid
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
            #print(f"{upTrain[-2].shape} -> {upTrain[-1].shape} ({dwTrain[-2-level].shape})")
        res = self.lastTouch(torch.cat( (upTrain[-1], images ), dim=1 ))
        return res * self.amplitude + images[:,[0],...]

    def forward(self, images):
        if self.inChannels > 1 and  images.shape[1] == 1 : # fill missing channels with noise
            images = images.repeat((1,self.inChannels,1,1))
            torch.nn.init.normal_( images[:,1:,:,:] , mean=0.0, std=0.1 )
        with torch.no_grad():
            images = images.clone().detach()
            images[:,[0],*self.cfg.gapRng] = self.preFill(images)
            images, norms = normalizeImages(images)
        results = self.dropIN(images)
        return reNormalizeImages(results, norms)








class GeneratorTemplate(SubGeneratorTemplate):

    def __init__(self, gapW, batchNorm=True, inChannels=3):
        super(GeneratorTemplate, self).__init__(gapW, False, batchNorm, inChannels=inChannels)
        self.brickGenerator = None


    def __generatePatches(self, images, noises=None) :
        if noises is None :
            noises = torch.randn( 1 if images.dim() < 3 else images.shape[0], TCfg.latentDim).to(TCfg.device)
        return self.forward((images,noises))


    def __fillImages(self, images, noises=None) :
        images[self.cfg.gapRng] = self.generatePatches(images, noises)
        return images


    def __generateImages(self, images, noises=None) :
        clone = images.clone()
        return self.fillImages(clone, noises)


    def forward(self, images):
        # channel 0
        with torch.no_grad():
            images = images.clone().detach()
            images[:,[0],*self.cfg.gapRng] = self.preFill(images)
            images, norms = normalizeImages(images)
        # channel 1
        stripeImages = self.stripeGenerator.forward(images)
        # channel 2
        bricksIn = torch.cat((      images.view(-1,1, *self.brickGenerator.cfg.sinoSh),
                              stripeImages.view(-1,1, *self.brickGenerator.cfg.sinoSh)),
                             dim=1)
        bricksM = self.brickGenerator.forward(bricksIn).view(-1,1, *self.cfg.sinoSh)
        # channel 3
        edge = self.brickGenerator.cfg.sinoSh[-2]//2
        bricksIn = torch.cat((      images[:,:,edge:-edge,:].reshape(-1,1, *self.brickGenerator.cfg.sinoSh),
                              stripeImages[:,:,edge:-edge,:].reshape(-1,1, *self.brickGenerator.cfg.sinoSh)),
                             dim=1)
        bricksP = images.clone()
        bricksP[:,:,edge:-edge,:] = self.brickGenerator.forward(bricksIn) \
            .view(-1,1, self.cfg.sinoSh[-2]-2*edge, self.cfg.sinoSh[-1] )
        # combine channels and drop in
        modelIn = torch.cat((images, stripeImages, bricksM, bricksP), dim=1)
        results = self.dropIN(modelIn)
        return reNormalizeImages(results, norms)


    ### this version of forward is only to calculate what bricks generator does with no main generator.
    def __forward(self, images):
        # channel 0
        with torch.no_grad():
            images = images.clone().detach()
            images[:,[0],*self.cfg.gapRng] = self.preFill(images)
            images, norms = normalizeImages(images)
        bricksM = self.brickGenerator.forward(images.view(-1,1, *self.brickGenerator.cfg.sinoSh))
        bricksM = bricksM.view(-1,1, *self.cfg.sinoSh)
        edge = self.brickGenerator.cfg.sinoSh[-2]//2
        bricksP = images.clone()
        bricksP[:,:,edge:-edge,:] = self.brickGenerator.forward(images[:,:,edge:-edge,:].reshape(-1,1, *self.brickGenerator.cfg.sinoSh)) \
            .view(-1,1, self.cfg.sinoSh[-2]-2*edge, self.cfg.sinoSh[-1] )
        results = ( bricksM + bricksM ) / 2
        return reNormalizeImages(results, norms)

    ### this version of forward is to train only stripe generator with no main generator.
    ### to be used with specific transformGTforStripe - see it below
    def __forward(self, images):
        with torch.no_grad():
            images = images.clone().detach()
            images[:,[0],*self.cfg.gapRng] = self.preFill(images)
            images, norms = normalizeImages(images)
        stripeImages = self.stripeGenerator.forward(images)
        return reNormalizeImages(stripeImages, norms)

def transformGT_forStripeTraining(images):
    with torch.no_grad():
        images, orgdims = unsqeeze4dim(images)
        images = torch.nn.functional.interpolate(images, scale_factor=(1/DCfg.gapW,1), mode='bilinear')
        images = torch.nn.functional.interpolate(images, scale_factor=(  DCfg.gapW,1), mode='bilinear')
    return squeezeOrg(images, orgdims)





generator = initIfNew('generator')
lowResGenerators = initIfNew('lowResGenerators', {})


class DiscriminatorTemplate(nn.Module):

    def __init__(self, omitEdges=0):
        super(DiscriminatorTemplate, self).__init__()
        self.baseChannels = 64
        self.omitEdges = omitEdges


    def encblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True,
                                padding='same', padding_mode='reflect') \
                                if stride == 1 and dopadding else \
                                nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True)
                     )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)

    def createHead(self) :
        encSh = self.body(torch.zeros((1,1,*DCfg.sinoSh))).shape
        linChannels = math.prod(encSh)
        toRet = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(0.4),
            nn.Linear(linChannels, self.baseChannels*4),
            #nn.Linear(linChannels, 1),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.4),
            nn.Linear(self.baseChannels*4, 1),
            nn.Sigmoid(),
        )
        fillWheights(toRet)
        return toRet

    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if self.omitEdges :
            images = images.clone() # I want to exclude two blocks on the edges :
            images[ ..., :self.omitEdges, DCfg.gapRngX ] = 0
            images[ ..., -self.omitEdges:, DCfg.gapRngX ] = 0
        convRes = self.body(images)
        res = self.head(convRes)
        return res

discriminator = initIfNew('discriminator')


def createOptimizer(model, lr) :
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.5, 0.999)
    )
optimizer_G = initIfNew('optimizer_G')
optimizer_D = initIfNew('optimizer_D')
scheduler_G = initIfNew('scheduler_G')
scheduler_D = initIfNew('scheduler_D')

def adjustScheduler(scheduler, iniLr, target) :
    if scheduler is None :
        return ""
    gamma = scheduler.gamma
    curLR = scheduler.get_last_lr()[0] / iniLr
    if gamma < 1 and curLR > target \
    or gamma > 1 and curLR < target :
        scheduler.step()
    return f"LR : {scheduler.get_last_lr()[0]:.3e} ({curLR:.3e}). "


def restoreCheckpoint(path=None, logDir=None) :
    if logDir is None :
        logDir = TCfg.logDir
    if path is None :
        if os.path.exists(logDir) :
            raise Exception(f"Starting new experiment with existing log directory \"{logDir}\"."
                            " Remove it .")
        try : os.remove(TCfg.historyHDF)
        except : pass
        return 0, 0, 0, None, 0, TrainResClass()
    else :
        return loadCheckPoint(path, generator, discriminator, optimizer_G, optimizer_D)


def saveModels(path="") :
    save_model(generator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_gen.pt" )
    save_model(discriminator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_dis.pt"  )



BCE = nn.BCELoss(reduction='none')
def loss_Adv(images, truth):
    labels = torch.full((images.shape[0], 1),  (1 - TCfg.labelSmoothFac ) if truth else TCfg.labelSmoothFac,
                dtype=torch.float, device=TCfg.device, requires_grad=False)
    predictions = discriminator(images)
    return BCE(predictions, labels), predictions

def loss_Adv_Gen(p_true, p_pred):
    return loss_Adv(p_pred, True)[0]

def loss_Adv_Dis(p_true, p_pred):
    loss_true, predictions_true = loss_Adv(p_true, True)
    loss_pred, predictions_pred = loss_Adv(p_pred, False)
    return ( torch.cat((loss_true, loss_pred), dim=0),
             torch.cat((predictions_true, predictions_pred), dim=0) )


MSE = nn.MSELoss(reduction='none')
def loss_MSE(p_true, p_pred):
    return MSE(p_true[DCfg.gapRng], p_pred[DCfg.gapRng]).sum(dim=(-1,-2,-3))

def loss_MSEN(p_true, p_pred):
    rawLoss = loss_MSE(p_true, p_pred)
    stds = 1e-7 +  calculateNorm(p_true)[0].view([-1])
    return rawLoss / stds**2

def loss_SMSE(p_true, p_pred):
    mseLoss = MSE(p_true[DCfg.gapRng], p_pred[DCfg.gapRng])
    return (torch.square(mseLoss)).sum(dim=(-1,-2,-3))

L1L = nn.L1Loss(reduction='none')
def loss_L1L(p_true, p_pred):
    return L1L(p_true[DCfg.gapRng], p_pred[DCfg.gapRng]).sum(dim=(-1,-2,-3))

def loss_L1LN(p_true, p_pred):
    rawLoss = loss_L1L(p_true, p_pred)
    stds = 1e-7 + calculateNorm(p_true)[0].view([-1])
    return rawLoss / stds

SSIM = ssim.SSIM(data_range=2.0, size_average=False, channel=1, win_size=1)
def loss_SSIM(p_true, p_pred):
    p_true, _ = unsqeeze4dim(p_true)
    p_pred, _ = unsqeeze4dim(p_pred)
    #return (1 - SSIM( p_true+0.5, p_pred+0.5 ) ) / 2
    return (1 - SSIM( p_true, p_pred ) ) / 2

MSSSIM = ssim.MS_SSIM(data_range=2.0, size_average=False, channel=1, win_size=1)
def loss_MSSSIM(p_true, p_pred):
    p_true, _ = unsqeeze4dim(p_true)
    p_pred, _ = unsqeeze4dim(p_pred)
    #return (1 - MSSSIM( p_true+0.5, p_pred+0.5 ) ) / 2
    return (1 - MSSSIM( p_true, p_pred ) ) / 2


def loss_COR(p_true, p_pred):
    d_true, _ = unsqeeze4dim(p_true[DCfg.gapRng])
    d_pred, _ = unsqeeze4dim(p_pred[DCfg.gapRng])
    means = torch.mean(d_true, dim=(-1,-2), keepdim=True)
    d_true = d_true - means
    d_pred = d_pred - means
    cor = (d_true*d_pred).sum(dim=(-1,-2))
    dist = 1 - cor / (d_true**2).sum(dim=(-1,-2))
    return torch.abs(dist.squeeze(1))
    #return 1 - cor / torch.sqrt( ( (p_true-means)**2 ).sum(dim=(-1,-2)) * ( (p_pred-means)**2 ).sum(dim=(-1,-2)) + 1e-7 )

def loss_STD(p_true, p_pred):
    p_true, _ = unsqeeze4dim(p_true[DCfg.gapRng])
    p_pred, _ = unsqeeze4dim(p_pred[DCfg.gapRng])
    return (p_true - p_pred).std(dim=(-1,-2,-3))

normalHist = torch.tensor([0.044058, 0.09185, 0.14988, 0.19145, 0.19146, 0.14987, 0.09184, 0.04405])
def loss_HIST(p_true, p_pred):
    global normalHist
    diffImages = p_true[DCfg.gapRng] - p_pred[DCfg.gapRng]
    normDiff, _ = normalizeImages(diffImages)
    normDiff = normDiff.view(normDiff.shape[0],-1)
    normalHist = normalHist.to(normDiff.device)
    toRet = []
    for imgn in range(normDiff.shape[0]) :
        img = normDiff[imgn,...]
        hist = torch.histc(img, bins=8, min=-2, max=2) / math.prod(img.shape)
        img_loss = torch.sum( (hist - normalHist)**2 )
        toRet.append( img_loss )
    return torch.stack(toRet)

@dataclass
class Metrics:
    calculate : callable
    norm : float # normalization factor - result of calculate on untrained model output
    weight : float # weight in the final loss function; zero means no loss contribution

metrices = {
    'Adv'    : Metrics(loss_Adv_Gen, 0, 0),
    'MSE'    : Metrics(loss_MSE,     1, 1),
    #'SMSE'   : Metrics(loss_SMSE,    1, 1),
    'MSEN'   : Metrics(loss_MSEN,    1, 1),
    'L1L'    : Metrics(loss_L1L,     1, 1),
    'L1LN'   : Metrics(loss_L1LN,    1, 1),
    'SSIM'   : Metrics(loss_SSIM,    1, 1),
    'MSSSIM' : Metrics(loss_MSSSIM,  1, 1),
    'STD'    : Metrics(loss_STD,     1, 1),
    'COR'    : Metrics(loss_COR,     1, 1),
    'HIST'   : Metrics(loss_HIST,    1, 1),
}

# Gap 2 metrices
{
#metrices = {
#    'Adv'    : Metrics(loss_Adv_Gen, 0,         0),
#    'MSE'    : Metrics(loss_MSE,     1.154e-01, 1),
#    'L1L'    : Metrics(loss_L1L,     2.571e+00, 1),
#    'SSIM'   : Metrics(loss_SSIM,    4.183e-04, 1),
#    'MSSSIM' : Metrics(loss_MSSSIM,  4.515e-06, 1),
#}
#metricesTrain = {
#    'Adv'    : Metrics(loss_Adv_Gen, 0,         0),
#    'MSE'    : Metrics(loss_MSE,     5.836e-01, 1),
#    'L1L'    : Metrics(loss_L1L,     9.742e+00, 1),
#    'SSIM'   : Metrics(loss_SSIM,    8.717e-04, 1),
#    'MSSSIM' : Metrics(loss_MSSSIM,  3.358e-05, 1),
#}
}


minMetrices = None
maxMetrices = None

def trackExtremes(track=True) :
    global minMetrices, maxMetrices
    toRet = (minMetrices, maxMetrices)
    if not track :
        minMetrices = None
        maxMetrices = None
    else :
        minMetrices = { key: None for key in metrices.keys() }
        minMetrices['loss'] = None
        maxMetrices = { key: None for key in metrices.keys() }
        maxMetrices['loss'] = None
    return toRet

def updateExtremes(vector, key, p_true, p_pred) :
    global minMetrices, maxMetrices
    if maxMetrices is not None :
        pos = vector.argmax()
        if maxMetrices[key] is None or vector[pos] > maxMetrices[key][0] :
            maxMetrices[key] = (vector[pos].item(),
                                p_true[pos,...].clone().detach(),
                                p_pred[pos,...].clone().detach() )
            #if key == 'loss' :
            #    print(pos.item())
            #    plotImage(maxMetrices['loss'][1][0].clone().detach().transpose(-1,-2).cpu().numpy())
            #    print()
    if minMetrices is not None :
        pos = vector.argmin()
        if minMetrices[key] is None or vector[pos] < minMetrices[key][0] :
            minMetrices[key] = (vector[pos].item(),
                                p_true[pos,...].clone().detach(),
                                p_pred[pos,...].clone().detach() )



def transformGT(images):
    return images


def loss_Gen(p_true, p_pred):
    global metrices, minMetrices, maxMetrices
    loss = 0
    sumweights = 0
    individualLosses = {}
    p_true = transformGT(p_true)
    for key, metrics in metrices.items():
        if metrics.norm > 0 :
            with torch.set_grad_enabled( metrics.weight > 0 ) :
                thisLoss = metrics.calculate(p_true, p_pred) / metrics.norm
                individualLosses[key] = thisLoss.sum().item()
                updateExtremes(thisLoss, key, p_true, p_pred)
                if  metrics.weight > 0 :
                    loss += metrics.weight * thisLoss
                    sumweights += metrics.weight
        else :
            individualLosses[key] = p_true.shape[0]
    loss /= sumweights
    updateExtremes(loss, 'loss', p_true, p_pred)
    return loss.sum() , individualLosses

def loss_Dis(p_true, p_pred):
    p_true = transformGT(p_true)
    return loss_Adv_Dis(p_true, p_pred)




@dataclass(repr = False)
class TrainResClass:
    lossD : float = 0
    lossG : float = 0
    metrices : dict = field(default_factory=lambda: { key: 0 for key in metrices.keys() })
    predReal : float = 0
    #predPre : float = 0
    predFake : float = 0
    predGen : float = 0
    nofIm : int = 0

    def __repr__(self):
        if self.nofIm == 0 :
            return "No train results yet accumulated."
        return f"Images: {self.nofIm}. DIS: {self.lossD/self.nofIm:.3e}, GEN: {self.lossG/self.nofIm:.3e}. " + \
               f"Probs: True {self.predReal/self.nofIm:.3e}, Fake {self.predFake/self.nofIm:.3e}.\n" + \
               f"Individual losses: " + ' '.join([ f"{key}: {self.metrices[key]/self.nofIm:.3e} "
                                                   for key in self.metrices.keys()  ] ) +\
                "\n"

    def __add__(self, other):
        toRet = TrainResClass()
        for field in dataclasses.fields(TrainResClass):
            fn = field.name
            if fn == 'metrices' :
                for key in self.metrices.keys() :
                    toRet.metrices[key] = self.metrices[key] + other.metrices[key]
            else :
                setattr(toRet, fn, getattr(self, fn) + getattr(other, fn) )
        return toRet

    def __mul__(self, other):
        toRet = TrainResClass()
        for field in dataclasses.fields(TrainResClass):
            fn = field.name
            if fn == 'metrices' :
                for key in self.metrices.keys() :
                    toRet.metrices[key] = self.metrices[key] * other
            else :
                setattr(toRet, fn, getattr(self, fn) * other)
        return toRet

    __rmul__ = __mul__


def summarizeMe(toSumm, onPrep=True):
    global generator, discriminator, metrices

    if isinstance(toSumm, torch.utils.data.Dataset) :
        loader = createDataLoader(toSumm)
        return summarizeMe(loader, onPrep)

    sumAcc = TrainResClass()

    #generator.train()
    generator.eval()
    #discriminator.eval()

    def summarizeImages(images) :
        nonlocal sumAcc

        images = unsqeeze4dim(images.to(TCfg.device))[0]
        nofIm = images.shape[0]
        sumAcc.nofIm += nofIm

        batchSplit = TCfg.batchSplit if TCfg.batchSplit > 1 else 1
        subBatchSize = nofIm // batchSplit
        fakeImages = images.clone()
        for i in range(batchSplit) :
            subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize]
            subImages = images[subRange,...]
            patchImages = generator.preFill(subImages) \
                            if onPrep else \
                          generator.forward(subImages)[DCfg.gapRng]
            fakeImages[subRange,...,DCfg.gapRngX] = patchImages

        genLoss, indLosses = loss_Gen(images, fakeImages)
        sumAcc.lossG += genLoss.item()
        for key in indLosses.keys() :
            sumAcc.metrices[key] += indLosses[key]

        if metrices['Adv'].weight > 0 : # discriminator
            disLoss, probs = loss_Dis(images, fakeImages)
            sumAcc.lossD += disLoss.item()
            sumAcc.predReal += probs[:nofIm,0].sum()
            sumAcc.predFake += probs[nofIm:,0].sum()

    with torch.no_grad() :
        if isinstance(toSumm, torch.utils.data.DataLoader) :
            for it , data in tqdm.tqdm(enumerate(toSumm), total=int(len(toSumm))):
                summarizeImages(data[0])
        elif isinstance(toSumm, torch.Tensor) :
            summarizeImages(toSumm)
        else :
            raise Exception(f"Unknown type of input to summarizeMe: {type(toSumm)}.")

    print(sumAcc)
    return sumAcc


def generateDisplay(inp=None, boxes=None) :
    #images = images.to(TCfg.device)
    images = refImages if inp is None else inp
    images, orgDim = unsqeeze4dim(images)
    nofIm = images.shape[0]
    viewLen = DCfg.sinoSh[-1]

    genImages = images.clone()
    preImages = images.clone()
    views = torch.empty((nofIm, 4, viewLen, viewLen ), dtype=torch.float32, device=TCfg.device)
    with torch.no_grad() :
        prePatches = generator.preFill(images)
        preImages[DCfg.gapRng] = prePatches
        genPatches = generator.forward(images)[DCfg.gapRng]
        genImages[DCfg.gapRng] = genPatches
    hGap = DCfg.gapW // 2

    if inp is None :
        boxes = refBoxes
    elif boxes is None : # find worst boxes
        diffImages = torch.abs(genImages - images)
        diffY = fn.conv2d(diffImages, torch.ones((1,1,diffImages.shape[-1],diffImages.shape[-1]), device=diffImages.device))
        boxes = diffY.squeeze(1,-1).argmax(dim=-1)

    for curim in range(nofIm) :
        rng = np.s_[curim, 0, boxes[curim] : boxes[curim] + viewLen, : ]
        views[curim,0,...] = images   [rng]
        views[curim,1,...] = preImages[rng]
        views[curim,2,...] = genImages[rng]
        views[curim,3,...] = 0
        views[curim,3,*DCfg.gapRng] = (genPatches - prePatches)[rng]
        views[curim,3,:,hGap:hGap+DCfg.gapW] = (images[DCfg.gapRng] - prePatches)[rng]
        views[curim,3,:,-DCfg.gapW-hGap:-hGap] = (images[DCfg.gapRng] - genPatches)[rng]
    return views, squeezeOrg(genImages, orgDim) , squeezeOrg(preImages, orgDim)


def displayImages(inp=None, boxes=None) :
    allImages = generateDisplay(inp, boxes)
    views, genImages, _ = allImages
    genImages = unsqeeze4dim(genImages)[0]
    views = views.cpu().numpy()
    nofIm = views.shape[0]
    for curim in range(nofIm) :
        if not DCfg.brick :
            plotImage(genImages[curim,0,...].transpose(-1,-2).cpu().numpy())
        vmin = views[curim,0:3,...].min()
        vmax = views[curim,0:3,...].max()
        plt.figure(frameon=False)
        for id in range(3) :
            plt.subplot(1, 4, id + 1)
            plt.imshow(views[curim,id,...], cmap='gray', vmin=vmin, vmax=vmax)
            plt.axis("off")
        vmm = max( -views[curim,3,...].min(), views[curim,3,...].max() )
        plt.subplot(1, 4, 4)
        plt.imshow(views[curim,3,...], cmap='gray', vmin=-vmm, vmax=vmm)
        plt.axis("off")
        plt.show()
    return allImages



def calculateWeights(images) :
    return None

def calculateNorm(images) :
    noGapImages = torch.cat( (images[...,:DCfg.gapRngX.start], images[...,DCfg.gapRngX.stop:]), dim=-1)
    toRet = torch.std_mean( noGapImages, dim=(-1,-2), keepdim=True )
    return toRet[0], toRet[1]


def saveCheckPoint(path, epoch, iterations, minGEpoch, lastGLoss,
                   generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None,
                   startFrom=0, interimRes=TrainResClass()) :
    checkPoint = {}
    checkPoint['epoch'] = epoch
    checkPoint['iterations'] = iterations
    checkPoint['minGEpoch'] = minGEpoch
    checkPoint['lastGLoss'] = lastGLoss
    checkPoint['startFrom'] = startFrom
    checkPoint['generator'] = generator.state_dict()
    checkPoint['discriminator'] = discriminator.state_dict()
    if not optimizerGen is None :
        checkPoint['optimizerGen'] = optimizerGen.state_dict()
    if not schedulerGen is None :
        checkPoint['schedulerGen'] = schedulerGen.state_dict()
    if not optimizerDis is None :
        checkPoint['optimizerDis'] = optimizerDis.state_dict()
    if not schedulerDis is None :
        checkPoint['schedulerDis'] = schedulerDis.state_dict()
    checkPoint['resAcc'] = interimRes
    torch.save(checkPoint, path)


def loadCheckPoint(path, generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None) :
    checkPoint = torch.load(path, map_location=TCfg.device, weights_only=False)
    epoch = checkPoint['epoch']
    iterations = checkPoint['iterations']
    minGEpoch = checkPoint['minGEpoch']
    lastGLoss = checkPoint['lastGLoss']
    startFrom = checkPoint['startFrom'] if 'startFrom' in checkPoint else 0
    generator.load_state_dict(checkPoint['generator'])
    discriminator.load_state_dict(checkPoint['discriminator'])
    if not optimizerGen is None :
        optimizerGen.load_state_dict(checkPoint['optimizerGen'])
    if not schedulerGen is None :
        schedulerGen.load_state_dict(checkPoint['schedulerGen'])
    if not optimizerDis is None :
        optimizerDis.load_state_dict(checkPoint['optimizerDis'])
    if not schedulerDis is None :
        schedulerDis.load_state_dict(checkPoint['schedulerDis'])
    interimRes = checkPoint['resAcc'] if 'resAcc' in checkPoint else TrainResClass()

    return epoch, iterations, minGEpoch, lastGLoss, startFrom, interimRes



skipDis = False

def train_step(allImages):
    global skipGen, skipDis

    trainRes = TrainResClass()
    allImages, _ = unsqeeze4dim(allImages.to(TCfg.device))
    nofAllIm = allImages.shape[0]

    while trainRes.nofIm < nofAllIm :

        images = allImages[ trainRes.nofIm : trainRes.nofIm + TCfg.batchSize , ... ]
        #fakeImages = images.clone().detach()
        nofIm = images.shape[0]
        trainRes.nofIm += nofIm
        #images, _ = unsqeeze4dim(images.to(TCfg.device))
        batchSplit = TCfg.batchSplit if TCfg.batchSplit > 1 else 1
        subBatchSize = nofIm // batchSplit

        # train discriminator
        if metrices['Adv'].weight > 0 :
            optimizer_D.zero_grad()
            for i in range(batchSplit) :
                subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize]
                subImages = images[subRange,...].clone().detach()
                with torch.no_grad() : # create fake images to be descriminated
                    subFakeImages = generator.generateImages(subImages)
                subImages.requires_grad = True
                subFakeImages.requires_grad = True
                disLoss, probs = loss_Dis(subImages, subFakeImages)
                trainRes.lossD += disLoss.item()
                trainRes.predReal += probs[:nofIm,0].sum().item()
                trainRes.predFake += probs[nofIm:,0].sum().item()
                disLoss /= subBatchSize
                disLoss.backward()
            optimizer_D.step()
            optimizer_D.zero_grad(set_to_none=True)


        # train generator
        optimizer_G.zero_grad(set_to_none=False)
        for i in range(batchSplit) :
            subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize]
            subImages = images[subRange,...]#.clone().detach()
            subImages.requires_grad = True
            subFakeImages = generator.generateImages(subImages)
            #subFakeImages = fakeImages[subRange,...]
            #subFakeImages[DCfg.gapRng] = generator.generatePatches(subImages)
            genLoss, indLosses = loss_Gen(subImages, subFakeImages)
            trainRes.lossG += genLoss.item()
            for key in indLosses.keys() :
                trainRes.metrices[key] += indLosses[key]
            genLoss /= subBatchSize
            genLoss.backward()
        optimizer_G.step()
        optimizer_G.zero_grad(set_to_none=True)

    return trainRes


epoch=initIfNew('epoch', 0)
iter = initIfNew('iter', 0)
imer = initIfNew('iter', 0)
minGEpoch = initIfNew('minGEpoch')
minGLoss = initIfNew('lastGLoss')
startFrom = initIfNew('startFrom', 0)

def beforeEachEpoch(locals) :
    return

def afterEachEpoch(locals) :
    return

def beforeReport(locals) :
    return

def afterReport(locals) :
    return

trainLoader=None
testLoader=None
resAcc = TrainResClass()

def train(savedCheckPoint):
    global epoch, minGLoss, minGEpoch, iter, startFrom, imer, resAcc
    global trainLoader, testLoader, testSet, refImages, minMetrices, maxMetrices
    lastGLoss = minGLoss

    discriminator.to(TCfg.device)
    generator.to(TCfg.device)
    lastUpdateTime = time.time()
    lastSaveTime = time.time()

    while TCfg.nofEpochs is None or epoch <= TCfg.nofEpochs :
        epoch += 1
        beforeEachEpoch(epoch)

        trainLoader = createDataLoader(trainSet, num_workers=TCfg.num_workers)
        testLoader = createDataLoader(testSet, num_workers=TCfg.num_workers)
        generator.train()
        discriminator.train()
        resAcc = TrainResClass()
        updAcc = TrainResClass()
        _ = trackExtremes()


        for it , data in tqdm.tqdm(enumerate(trainLoader), total=int(len(trainLoader))):
            if startFrom :
                startFrom -= 1
                continue
            iter += 1
            images = data[0].to(TCfg.device)
            imer += images.shape[0]
            trainRes = train_step(images)
            resAcc += trainRes
            updAcc += trainRes

            #if True:
            if time.time() - lastUpdateTime > 60  or imer == images.shape[0]:

                # generate previews
                generator.eval()
                extImages = torch.stack((maxMetrices['loss'][1],minMetrices['loss'][1]))
                extViews, extGen, _ = generateDisplay(extImages)
                extViews = extViews.cpu().numpy()
                refViews, genImages, _ = generateDisplay()
                refViews = refViews.cpu().numpy()
                rndIndeces = random.sample(range(images.shape[0]), 2)
                rndViews, rndGen, _ = generateDisplay(images[rndIndeces,...])
                rndViews = rndViews.cpu().numpy()
                generator.train()


                IPython.display.clear_output(wait=True)
                beforeReport(locals())
                print(f"Epoch: {epoch} ({minGEpoch}). ", end=' ')
                print(updAcc)
                updAcc *= 1/updAcc.nofIm
                writer.add_scalars("Losses per iter",
                                   {'Dis': updAcc.lossD
                                   ,'Gen': updAcc.lossG
                                   #,'Adv' : updAcc.metrices['Adv']
                                   }, imer )
                for key in updAcc.metrices.keys() :
                    if metrices[key].norm > 0 :
                        writer.add_scalars("Metrices per iter", {key : updAcc.metrices[key],}, imer )
                #writer.add_scalars("Metrices per iter", updAcc.metrices, imer )
                writer.add_scalars("Probs per iter",
                                   {'Ref':updAcc.predReal
                                   ,'Gen':updAcc.predFake
                                   #,'Pre':trainRes.predGen
                                   }, imer )

                subLay = (1,1)
                def addSubplot(pos, img, sym=True) :
                    nonlocal subLay
                    plt.subplot(*subLay, pos) ;
                    vmm = max( -img.min(), img.max() )
                    plt.imshow(img, cmap='gray', vmin = -vmm if sym else None,
                                                 vmax =  vmm if sym else None)
                    plt.axis("off")

                if not DCfg.brick :
                    subLay = (3,1)
                    plt.figure(frameon=True, layout='compressed', facecolor=(0,0.3,0.5))
                    #plt.subplots_adjust(hspace=0.5, wspace=0)
                    addSubplot(1, rndGen[0,0,...].transpose(-1,-2).cpu().numpy(), False)
                    addSubplot(2, genImages[0,0,...].transpose(-1,-2).cpu().numpy(), False)
                    addSubplot(3, extGen[0,0,...].transpose(-1,-2).cpu().numpy(), False)
                    plt.show()

                subLay = (2,5)
                plt.figure(frameon=True, layout='compressed', facecolor=(0,0.3,0.5))
                #plt.subplots_adjust(hspace = 0.1, wspace=0.1)
                addSubplot( 1, extViews[0,2],False)
                addSubplot( 2, rndViews[0,0],False)
                addSubplot( 3, rndViews[0,1],False)
                addSubplot( 4, refViews[3,3])
                addSubplot( 5, refViews[2,3])
                addSubplot( 6, extViews[0,3])
                addSubplot( 7, rndViews[0,2],False)
                addSubplot( 8, rndViews[0,3])
                addSubplot( 9, refViews[1,3])
                addSubplot(10, refViews[0,3])
                plt.show()


                afterReport(locals())
                lastUpdateTime = time.time()
                updAcc = TrainResClass()
                _ = trackExtremes()

            if time.time() - lastSaveTime > 3600 :
                lastSaveTime = time.time()
                saveCheckPoint(savedCheckPoint+"_hourly.pth",
                               epoch-1, imer, minGEpoch, minGLoss,
                               generator, discriminator,
                               optimizer_G, optimizer_D,
                               startFrom=it, interimRes=resAcc)
                saveModels(f"model_{TCfg.exec}_hourly")


        print(resAcc)
        resAcc *= 1/resAcc.nofIm
        writer.add_scalars("Losses per epoch",
                           {'Dis': resAcc.lossD
                           ,'Gen': resAcc.lossG
                           #,'Adv' : resAcc.metrices['Adv']
                           }, epoch )
        #writer.add_scalars("Metrices per epoch", resAcc.metrices, epoch )
        for key in resAcc.metrices.keys() :
            if metrices[key].norm > 0 :
                writer.add_scalars("Metrices per epoch", {key : resAcc.metrices[key],}, epoch )
        writer.add_scalars("Probs per epoch",
                           {'Ref':resAcc.predReal
                           ,'Gen':resAcc.predFake
                           #,'Pre':trainRes.predGen
                           }, epoch )


        generator.eval()
        displayImages()
        resTest = summarizeMe(testLoader, False)
        resTest *= 1/resTest.nofIm
        writer.add_scalars("Losses epoch test",
            {'Dis': resTest.lossD
            ,'Gen': resTest.lossG
            #,'Adv' : resTest.metrices['Adv']
            }, epoch )
        for key in resTest.metrices.keys() :
            if metrices[key].norm > 0 :
                writer.add_scalars("Metrices epoch test", {key : resTest.metrices[key],}, epoch )
        writer.add_scalars("Probs epoch test",
            {'Ref':resTest.predReal
            ,'Gen':resTest.predFake
            #,'Pre':trainRes.predGen
            }, epoch )
        generator.train()


        lastGLoss = resTest.lossG # Rec_test
        if minGLoss is None or minGLoss == 0 or lastGLoss < minGLoss :
            minGLoss = lastGLoss
            minGEpoch = epoch
        saveModels()
        os.system(f"mv {savedCheckPoint}.pth {savedCheckPoint}_previous.pth")
        saveCheckPoint(savedCheckPoint+".pth",
                       epoch, imer, minGEpoch, minGLoss,
                       generator, discriminator,
                       optimizer_G, optimizer_D)
        if minGEpoch == epoch :
            os.system(f"cp {savedCheckPoint}.pth {savedCheckPoint}_best.pth")
            os.system(f"cp {savedCheckPoint}_previous.pth {savedCheckPoint}_beforebest.pth")
            #saveModels(f"model_{TCfg.exec}_B")

        resAcc = TrainResClass()
        afterEachEpoch(locals())
        print("Epoch completed.\n")






def freeGPUmem() :
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()







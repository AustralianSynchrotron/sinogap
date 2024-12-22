
import IPython

import sys
import os
import random
import time
import gc
from dataclasses import dataclass, field
from enum import Enum

import math
import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torchvision
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import h5py
import tifffile
import tqdm


def initToNone(var) :
    if var in locals() :
        return locals(var)
    if var in globals() :
        return globals(var)
    return None


@dataclass
class TCfgClass:
    exec : int
    latentDim: int
    batchSize: int
    labelSmoothFac: float
    learningRateD: float
    learningRateG: float
    device: torch.device = torch.device('cpu')
    nofEpochs: int = 0
    historyHDF : str = field(repr = True, init = False)
    logDir : str = field(repr = True, init = False)
    def __post_init__(self):
        if self.device == torch.device('cpu')  :
            self.device = torch.device(f"cuda:{self.exec}")
        self.historyHDF = f"train_{self.exec}.hdf"
        self.logDir = f"runs/experiment_{self.exec}"
TCfg = initToNone('TCfg')


@dataclass
class DCfgClass:
    gapW : int
    sinoSh : tuple = field(repr = True, init = False)
    readSh : tuple = field(repr = True, init = False)
    sinoSize : int = field(repr = True, init = False)
    gapSh : tuple = field(repr = True, init = False)
    gapSize : int = field(repr = True, init = False)
    gapRngX : type(np.s_[:]) = field(repr = True, init = False)
    gapRng : type(np.s_[:]) = field(repr = True, init = False)
    disRng : type(np.s_[:]) = field(repr = True, init = False)
    def __post_init__(self):
        self.sinoSh = (5*self.gapW,5*self.gapW)
        self.readSh = (80, 80)
        self.sinoSize = math.prod(self.sinoSh)
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapSize = math.prod(self.gapSh)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[...,self.gapRngX]
        self.disRng = np.s_[ self.gapW:-self.gapW , self.gapRngX ]
DCfg = initToNone('DCfg')


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

    plt.style.use('default')
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


def plotImage(image) :
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()


def plotImages(images) :
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


def fillWheights(seq) :
    for wh in seq :
        if hasattr(wh, 'weight') :
            #torch.nn.init.xavier_uniform_(wh.weight)
            #torch.nn.init.zeros_(wh.weight)
            #torch.nn.init.constant_(wh.weight, 0)
            #torch.nn.init.uniform_(wh.weight, a=0.0, b=1.0, generator=None)
            torch.nn.init.normal_(wh.weight, mean=0.0, std=0.01)
        if hasattr(wh, 'bias') :
            torch.nn.init.normal_(wh.bias, mean=0.0, std=0.01)


def unsqeeze4dim(tens):
    orgDims = tens.dim()
    if tens.dim() == 2 :
        tens = tens.unsqueeze(0)
    if tens.dim() == 3 :
        tens = tens.unsqueeze(1)
    return tens, orgDims


def squeezeOrg(tens, orgDims):
    if orgDims == tens.dim():
        return tens
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


def saveCheckPoint(path, epoch, iterations, minGEpoch, minGdLoss,
                   generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None ) :
    checkPoint = {}
    checkPoint['epoch'] = epoch
    checkPoint['iterations'] = iterations
    checkPoint['minGEpoch'] = minGEpoch
    checkPoint['minGdLoss'] = minGdLoss
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
    torch.save(checkPoint, path)


def loadCheckPoint(path, generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None) :
    checkPoint = torch.load(path, map_location=TCfg.device)
    epoch = checkPoint['epoch']
    iterations = checkPoint['iterations']
    minGEpoch = checkPoint['minGEpoch']
    minGdLoss = checkPoint['minGdLoss']
    realModel(generator).load_state_dict(checkPoint['generator'])
    realModel(discriminator).load_state_dict(checkPoint['discriminator'])
    if not optimizerGen is None :
        optimizerGen.load_state_dict(checkPoint['optimizerGen'])
    if not schedulerGen is None :
        schedulerGen.load_state_dict(checkPoint['schedulerGen'])
    if not optimizerDis is None :
        optimizerDis.load_state_dict(checkPoint['optimizerDis'])
    if not schedulerDis is None :
        schedulerDis.load_state_dict(checkPoint['schedulerDis'])
    return epoch, iterations, minGEpoch, minGdLoss


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


def createWriter(logDir, addToExisting=False) :
    if not addToExisting and os.path.exists(logDir) :
        raise Exception(f"Log directory \"{logDir}\" for the experiment already exists."
                        " Remove it or implicitry overwrite with setting addToExisting to True.")
    return SummaryWriter(logDir)
writer = initToNone('writer')


class StripesFromHDF :

    def __init__(self, sampleName, maskName, bgName=None, dfName=None, loadToMem=True):

        sampleHDF = sampleName.split(':')
        if len(sampleHDF) != 2 :
            raise Exception(f"String \"{sampleName}\" does not represent an HDF5 format.")
        with h5py.File(sampleHDF[0],'r') as trgH5F:
            if  sampleHDF[1] not in trgH5F.keys():
                raise Exception(f"No dataset '{sampleHDF[1]}' in input file {sampleHDF[0]}.")
            self.data = trgH5F[sampleHDF[1]]
            if not self.data.size :
                raise Exception(f"Container \"{sampleName}\" is zero size.")
            self.sh = self.data.shape
            if len(self.sh) != 3 :
                raise Exception(f"Dimensions of the container \"{sampleName}\" is not 3 {self.sh}.")
            self.fsh = self.sh[1:3]
            self.volume = None
            if loadToMem :
                self.volume = np.empty(self.sh, dtype=np.float32)
                self.data.read_direct(self.volume)
                trgH5F.close()

            def loadImage(imageName) :
                if not imageName:
                    return None
                imdata = imread(imageName).astype(np.float32)
                if len(imdata.shape) == 3 :
                    imdata = np.mean(imdata[:,:,0:3], 2)
                if imdata.shape != self.fsh :
                    raise Exception(f"Dimensions of the input image \"{imageName}\" {imdata.shape} "
                                    f"do not match the face of the container \"{sampleName}\" {self.fsh}.")
                return imdata

            self.mask = loadImage(maskName)
            if self.mask is None :
                self.mask = np.ones(self.fsh, dtype=np.uint8)
            self.mask = self.mask.astype(bool)
            self.bg = loadImage(bgName)
            self.df = loadImage(dfName)
            if self.bg is not None :
                if self.df is not None:
                    self.bg -= self.df
                self.mask  &=  self.bg > 0.0

            self.allIndices = []
            for yCr in range(0,self.fsh[0]) :
                for xCr in range(0,self.fsh[1]) :
                    idx = np.s_[yCr,xCr]
                    if self.mask[idx] :
                        if self.volume is not None :
                            if self.df is not None :
                                self.volume[:,*idx] -= self.df[idx]
                            if self.bg is not None :
                                self.volume[:,*idx] /= self.bg[idx]
                        if  xCr + DCfg.readSh[1] < self.fsh[1] \
                        and np.all( self.mask[yCr,xCr+1:xCr+DCfg.readSh[1]] ) :
                            self.allIndices.append(idx)

    def get_dataset(self, transform=None) :

        class Sinos(torch.utils.data.Dataset) :

            def __init__(self, root, transform=None):
                self.container = root
                self.transform = transforms.Compose([transforms.ToTensor(), transform]) \
                    if transform else transforms.ToTensor()

            def __len__(self):
                return len(self.container.allIndices)

            def __getitem__(self, index=None, idxs=None, doTransform=True):
                if idxs is None or index is None :
                    idxs = random.randint(0,self.container.sh[0]-DCfg.readSh[0]-1)
                    index = random.randint(0,len(self.container.allIndices)-1)
                idx = self.container.allIndices[index]
                xyrng=np.s_[ idx[0], idx[1]:idx[1]+DCfg.readSh[1] ]
                if self.container.volume is not None :
                    data = self.container.volume[idxs:idxs+DCfg.readSh[0], *xyrng]
                else :
                    data = self.container.data[idxs:idxs+DCfg.readSh[0], *xyrng]
                    if self.container.df is not None :
                        data -= self.container.df[None,*xyrng]
                    if self.container.bg is not None :
                        data /= self.container.bg[None,*xyrng]
                if doTransform and self.transform :
                    data = self.transform(data)
                return (data, index, idxs)

        return Sinos(self, transform)


class StripesFromHDFs :
    def __init__(self, bases):
        self.collection = []
        for base in bases :
            self.collection.append(
                StripesFromHDF(f"storage/{base}.hdf:/data", f"storage/{base}.mask.tif", None, None) )
            print("Loaded set " + base)

    def get_dataset(self, transform=None) :

        class Sinos(torch.utils.data.Dataset) :

            def __init__(self, root, transform=None):
                self.container = root
                self.transform = transforms.Compose([transforms.ToTensor(), transform]) \
                    if transform else transforms.ToTensor()

            def __len__(self):
                return sum( [ len(set.allIndices) for set in self.container.collection ] )

            def __getitem__(self, index=None, idxs=None, doTransform=True):
                if idxs is None or index is None :
                    index = random.randint(0,len(self)-1)
                leftover = index
                useset = None
                for set in self.container.collection :
                    setLen = len(set.allIndices)
                    if leftover >= setLen :
                        leftover -= setLen
                    else :
                        useset = set
                        break
                if useset is None :
                    raise f"No set for index {index}."
                if idxs is None :
                    idxs = random.randint(0,useset.sh[0]-DCfg.readSh[0]-1)
                idx = useset.allIndices[leftover]
                xyrng=np.s_[ idx[0], idx[1]:idx[1]+DCfg.readSh[1] ]
                if useset.volume is not None :
                    data = useset.volume[idxs:idxs+DCfg.readSh[0], *xyrng]
                else :
                    data = useset.data[idxs:idxs+DCfg.readSh[0], *xyrng]
                    if useset.df is not None :
                        data -= useset.df[None,*xyrng]
                    if useset.bg is not None :
                        data /= useset.bg[None,*xyrng]
                if doTransform and self.transform :
                    data = self.transform(data)
                return (data, index, idxs)

        return Sinos(self, transform)


examplesDb = {}
examplesDb[2] = [(1271101, 570)
                ,(3007773,2063)
                ,(1271426,1140)
                ,(176088,893)
                ,(173107,273)
                ,(141881,817)
                ,(4076914,1642)
                ,(2880692,530)
                ,(1333420,160)
                ,(2997700,2321)
                ,(1385331,653)
                ,(132424,868)
                ,(1440290,204)
                ,(132352,757)
                ,(5053275,1700)
                ]
examplesDb[4] = [(1298309, 1015)
                #,(4612947, 2882)
                ,(2215760, 500)
                ,(2348095, 1684)
                ,(1907990, 1545)
                ,(291661, 724)
                ,(2489646, 1240)
                ,(1142687, 230)
                #,(974214, 631)
                ,(1429007,666)
                ,(152196,251)
                ,(1284589,62)
                ,(2963058,233)
                ,(200279,41)
                ,(1707893,914)
                ,(102151, 418)
                ]
examplesDb[8] = [(2348095, 1684)
                ,(1909160,333)
                ,(2489646, 1240)
                ,(1429010,666)
                ,(152196,251)
                ,(1707893,914)
                ,(102151, 418)]
examplesDb[16] = [ (2348095, 1684)
                 , (1958164,1391)
                 ,(1429010,666)
                 , (102151, 418)]
examples = initToNone('examples')

def createTrainSet() :
    listOfData = [ "4176862R_Eig_Threshold-4keV"
                 , "18515.Lamb1_Eiger_7m_45keV_360Scan"
                 , "23574.8965435L.Eiger.32kev_org"
                 , "23574.8965435L.Eiger.32kev_sft"
                 ]
    sinoRoot = StripesFromHDFs(listOfData)
    mytransforms = transforms.Compose([
            transforms.Resize(DCfg.sinoSh),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    return sinoRoot.get_dataset(mytransforms)


def createTrainLoader(trainSet, num_workers=os.cpu_count()) :
    return torch.utils.data.DataLoader(
        dataset=trainSet,
        batch_size=TCfg.batchSize,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )


class PrepackedHDF :

    def __init__(self, sampleName):
        sampleHDF = sampleName.split(':')
        if len(sampleHDF) != 2 :
            raise Exception(f"String \"{sampleName}\" does not represent an HDF5 format.")
        with h5py.File(sampleHDF[0],'r') as trgH5F:
            if  sampleHDF[1] not in trgH5F.keys():
                raise Exception(f"No dataset '{sampleHDF[1]}' in input file {sampleHDF[0]}.")
            self.data = trgH5F[sampleHDF[1]]
            if not self.data.size :
                raise Exception(f"Container \"{sampleName}\" is zero size.")
            self.sh = self.data.shape
            if len(self.sh) != 3 :
                raise Exception(f"Dimensions of the container \"{sampleName}\" is not 3 {self.sh}.")
            self.fsh = self.sh[1:3]
            if self.fsh != (80,80) :
                raise Exception(f"Dimensions of the container \"{sampleName}\" is not 80,80 {self.fsh}.")
            self.volume = None
            self.volume = np.empty(self.sh, dtype=np.float32)
            self.data.read_direct(self.volume)
            trgH5F.close()

    def get_dataset(self, transform=None) :

        class Sinos(torch.utils.data.Dataset) :

            def __init__(self, root, transform=None):
                self.container = root
                self.transform = transforms.Compose([transforms.ToTensor(), transform]) \
                    if transform else transforms.ToTensor()

            def __len__(self):
                return self.container.sh[0]

            def __getitem__(self, index):
                data=self.container.volume[[index],...]
                data = self.transform(data)
                return data

        return Sinos(self, transform)

def createTestSet() :
    sinoRoot = PrepackedHDF("storage/test/testSetSmall.hdf:/data")
    mytransforms = transforms.Compose([
            transforms.Resize(DCfg.sinoSh),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    return sinoRoot.get_dataset(mytransforms)


def createTestLoader(testSet, num_workers=os.cpu_count()) :
    return torch.utils.data.DataLoader(
        dataset=testSet,
        batch_size = TCfg.batchSize, # test requires no grad
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )



def createReferences(trainSet, toShow = 0) :
    global examples
    examples = examplesDb[DCfg.gapW].copy()
    if toShow :
        examples.insert(0, examples.pop(toShow))
    mytransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(DCfg.sinoSh),
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    refImages = torch.stack( [ mytransforms(trainSet.__getitem__(*ex, doTransform=False)[0])
                               for ex in examples ] ).to(TCfg.device)
    refNoises = torch.randn((refImages.shape[0],TCfg.latentDim)).to(TCfg.device)
    return refImages, refNoises
refImages = initToNone('refImages')
refNoises = initToNone('refNoises')


def showMe(trainSet, item=None) :
    global refImages, refNoises
    image = None
    if item is None :
        while True:
            image, index, idxs = trainSet[random.randint(0,len(trainSet)-1)]
            if image.mean() > 0 and image.min() < -0.1 :
                print (f"{index}, {idxs}")
                break
    elif isinstance(item, int) :
        image = refImages[0,...]
    else :
        image, _,_ = trainSet.__getitem__(*item)
    image = image.squeeze()
    tensorStat(image)
    plotImage(image.cpu())
    image = image.to(TCfg.device)


def realModel(mod) :
    return mod.module if isinstance(mod, nn.DataParallel) else mod


class GeneratorTemplate(nn.Module):

    def __init__(self, gapW):
        super(GeneratorTemplate, self).__init__()

        self.gapW = gapW
        self.sinoSh = (5*self.gapW,5*self.gapW) # 10,10
        self.sinoSize = math.prod(self.sinoSh)
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapSize = math.prod(self.gapSh)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[...,self.gapRngX]


    def encblock(self, chIn, chOut, kernel, stride=1, norm=True, dopadding=False) :
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


    def decblock(self, chIn, chOut, kernel, stride=1, norm=True, dopadding=False) :
        layers = []
        layers.append( nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True,
                                          padding=1) \
                       if stride == 1 and dopadding else \
                       nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True)
                      )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)


    def generatePatches(self, images, noises=None) :
        if noises is None :
            noises = torch.randn( 1 if images.dim() < 3 else images.shape[0], TCfg.latentDim).to(TCfg.device)
        return self.forward((images,noises))


    def fillImages(self, images, noises=None) :
        images[self.gapRng] = self.generatePatches(images, noises)
        return images


    def generateImages(self, images, noises=None) :
        clone = images.clone()
        return self.fillImages(clone, noises)

    def preProc(self, images) :
        images, orgDims = unsqeeze4dim(images)
        if self.gapW == 2:
            res = torch.zeros(images[self.gapRng].shape, device=images.device)
            res[...,0] += 2*images[...,self.gapRngX.start-1] + images[...,self.gapRngX.stop]
            res[...,1] += 2*images[...,self.gapRngX.stop] + images[...,self.gapRngX.start-1]
            res /= 3
        else :
            preImages = torch.nn.functional.interpolate(images, scale_factor=0.5, mode='area')
            res = None
            with torch.no_grad() :
                res = realModel(lowResGenerators[self.gapW//2]).generatePatches(preImages)
            res = torch.nn.functional.interpolate(res, scale_factor=2, mode='bilinear')
        return squeezeOrg(res, orgDims)


generator = initToNone('generator')
lowResGenerators = {}



class DiscriminatorTemplate(nn.Module):

    def __init__(self):
        super(DiscriminatorTemplate, self).__init__()

    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(1)
        convRes = self.body(images)
        res = self.head(convRes)
        return res
discriminator = initToNone('discriminator')


def createOptimizers() :
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=TCfg.learningRateG,
        betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=TCfg.learningRateD,
        betas=(0.5, 0.999)
    )
    return optimizer_G, optimizer_D
optimizer_G = initToNone('optimizer_G')
optimizer_D = initToNone('optimizer_D')


def restoreCheckpoint(path=None, logDir=None) :
    if logDir is None :
        logDir = TCfg.logDir
    if path is None :
        if os.path.exists(logDir) :
            raise Exception(f"Starting new experiment with existing log directory \"{logDir}\"."
                            " Remove it .")
        try : os.remove(TCfg.historyHDF)
        except : pass
        return 0, 0, 0, -1
    else :
        return loadCheckPoint(path, generator, discriminator, optimizer_G, optimizer_D)


def saveModels(path="") :
    save_model(generator, model_path=f"model_{TCfg.exec}_gen.pt")
    save_model(discriminator, model_path=f"model_{TCfg.exec}_dis.pt")


def createCriteria() :
    BCE = nn.BCELoss(reduction='none')
    MSE = nn.MSELoss(reduction='none')
    L1L = nn.L1Loss(reduction='none')
    return BCE, MSE, L1L
BCE, MSE, L1L = createCriteria()
lossDifCoef = 0

def applyWeights(inp, weights):
    inp = inp.squeeze()
    if not inp.dim() :
        inp = inp.unsqueeze(0)
    sum = len(inp)
    if not weights is None :
        inp *= weights
        sum = weights.sum()
    return inp.sum()/sum

def loss_Adv(y_true, y_pred, weights=None):
    loss = BCE(y_pred, y_true)
    return applyWeights(loss,weights)

def loss_MSE(p_true, p_pred, weights=None):
    loss = MSE(p_pred, p_true).mean(dim=(-1,-2))
    return applyWeights(loss,weights)

def loss_L1L(p_true, p_pred, weights=None):
    loss = L1L(p_pred, p_true).mean(dim=(-1,-2))
    return applyWeights(loss,weights)

eDinfo = None
def loss_Rec(p_true, p_pred, weights=None):
    global eDinfo
    loss = MSE(p_pred, p_true).mean(dim=(-1,-2)).squeeze()
    if loss.dim() :
        hDindex = loss.argmax()
        lDindex = loss.argmin()
        eDinfo = (hDindex, loss[hDindex].item(), lDindex, loss[lDindex].item() )
    return applyWeights(loss,weights)


def loss_Gen(y_true, y_pred, p_true, p_pred, weights=None):
    lossAdv = loss_Adv(y_true, y_pred, weights)
    lossDif = loss_Rec(p_pred, p_true)
    return lossAdv, lossDif


def summarizeSet(dataloader, onPrep=True):


    MSE_diffs, L1L_diffs, Rec_diffs = [], [], []
    totalNofIm = 0
    generator.to(TCfg.device)
    realModel(generator).eval()
    with torch.no_grad() :
        for it , data in tqdm.tqdm(enumerate(dataloader), total=int(len(dataloader))):
            images = data[0].squeeze(1).to(TCfg.device)
            nofIm = images.shape[0]
            totalNofIm += nofIm
            procImages, procData = imagesPreProc(images)
            prepImages = procImages.clone()
            if onPrep :
                prepImages[DCfg.gapRng] = realModel(generator).preProc(prepImages)
            else :
                prepImages[DCfg.gapRng] = realModel(generator).generatePatches(prepImages)
            procImages = imagesPostProc(prepImages, procData)
            MSE_diffs.append( nofIm * loss_MSE(images[DCfg.gapRng], procImages[DCfg.gapRng]))
            L1L_diffs.append( nofIm * loss_L1L(images[DCfg.gapRng], procImages[DCfg.gapRng]))
            Rec_diffs.append( nofIm * loss_Rec(images[DCfg.gapRng], procImages[DCfg.gapRng]))

    MSE_diff = sum(MSE_diffs) / totalNofIm
    L1L_diff = sum(L1L_diffs) / totalNofIm
    Rec_diff = sum(Rec_diffs) / totalNofIm
    print (f"Summary. Rec: {Rec_diff:.3e}, MSE: {MSE_diff:.3e}, L1L: {L1L_diff:.3e}.")
    return Rec_diff, MSE_diff, L1L_diff


def generateDiffImages(images, layout=None) :
    images, orgDim = unsqeeze4dim(images)
    dif = torch.zeros((images.shape[0], 1, *DCfg.sinoSh))
    hGap = DCfg.gapW // 2
    pre = images.clone()
    gen = images.clone()
    probs = None
    dists = None
    with torch.inference_mode() :
        generator.eval()
        pre[DCfg.gapRng] = realModel(generator).preProc(images)
        gen[DCfg.gapRng] = realModel(generator).generatePatches(images)
        dif[DCfg.gapRng] = (gen - pre)[DCfg.gapRng]
        dif[...,hGap:hGap+DCfg.gapW] = (images - pre)[DCfg.gapRng]
        dif[...,-DCfg.gapW-hGap:-hGap] = (images - gen)[DCfg.gapRng]
        for curim in range(images.shape[0]) :
            if ( cof := max(-dif[curim,...].min(),dif[curim,...].max()) ) != 0 :
                dif[curim,...] /= cof
            else :
                dif[curim,...] = 0
        probs = torch.empty(images.shape[0],3)
        dists = torch.empty(images.shape[0],3)
        discriminator.eval()
        probs[:,0] = discriminator(images)[:,0]
        probs[:,1] = discriminator(pre)[:,0]
        probs[:,2] = discriminator(gen)[:,0]
        dists[:,0] = loss_Rec(images[DCfg.gapRng], gen[DCfg.gapRng], calculateWeights(images))
        dists[:,1] = loss_MSE(images[DCfg.gapRng], gen[DCfg.gapRng])
        dists[:,2] = loss_L1L(images[DCfg.gapRng], gen[DCfg.gapRng])

    simages = None
    if not layout is None :
        def stretch(stretchme, mm, aa) :
            return ( stretchme - mm ) * 2 / aa - 1 if ampl > 0 else stretchme * 0
        simages = images.clone()
        for curim in range(images.shape[0]) :
            rng = np.s_[curim,...]
            minv = min(images[rng].min(), pre[rng].min(), gen[rng].min()).item()
            ampl = max(images[rng].max(), pre[rng].max(), gen[rng].max()).item() - minv
            simages[rng] = stretch(simages[rng], minv, ampl)
            pre[rng] = stretch(pre[rng], minv, ampl)
            gen[rng] = stretch(gen[rng], minv, ampl)

    cGap = DCfg.gapW
    collage = None
    if layout == 0 :
        collage = torch.empty(images.shape[0], 4, *DCfg.sinoSh)
        collage[:,0,...] = simages[:,0,...]
        collage[:,1,...] = pre[:,0,...]
        collage[:,2,...] = gen[:,0,...]
        collage[:,3,...] = dif[:,0,...]
    elif layout == 2 :
        collage = torch.zeros((images.shape[0], 1, DCfg.sinoSh[0]*2 + cGap, DCfg.sinoSh[1]*2 + cGap ))
        collage[..., :DCfg.sinoSh[0], :DCfg.sinoSh[1]] = gen
        collage[..., :DCfg.sinoSh[0], DCfg.sinoSh[1]+cGap:] = pre
        collage[..., DCfg.sinoSh[0]+cGap:, :DCfg.sinoSh[1]] = simages
        collage[..., DCfg.sinoSh[0]+cGap:, DCfg.sinoSh[1]+cGap:] = dif
    elif layout == 4 :
        collage = torch.zeros((images.shape[0], 1, DCfg.sinoSh[0], 4*DCfg.sinoSh[1] + 3*cGap))
        collage[..., :DCfg.sinoSh[1]] = simages
        collage[..., DCfg.sinoSh[1]+cGap:2*DCfg.sinoSh[1]+cGap] = gen
        collage[..., 2*DCfg.sinoSh[1]+2*cGap:3*DCfg.sinoSh[1]+2*cGap] = dif
        collage[..., 3*DCfg.sinoSh[1]+3*cGap:4*DCfg.sinoSh[1]+4*cGap] = pre
    elif layout == -4 :
        collage = torch.zeros( (images.shape[0], 1, 4*DCfg.sinoSh[0] + 3*cGap, DCfg.sinoSh[1]))
        collage[... , :DCfg.sinoSh[0] , : ] = simages
        collage[... , DCfg.sinoSh[0]+cGap:2*DCfg.sinoSh[0]+cGap , :] = gen
        collage[... , 2*DCfg.sinoSh[0]+2*cGap:3*DCfg.sinoSh[0]+2*cGap , : ] = dif
        collage[... , 3*DCfg.sinoSh[0]+3*cGap:4*DCfg.sinoSh[0]+4*cGap , : ] = pre
    else :
        collage = dif
    collage = squeezeOrg(collage,orgDim)

    return collage, probs, dists


def logStep(iter, write=True) :
    colImgs, probs, dists = generateDiffImages(refImages, layout=-4)
    probs = probs.mean(dim=0)
    dists = dists.mean(dim=0)
    colImgs = colImgs.squeeze()
    cSh = colImgs.shape
    gapH = DCfg.gapW
    collage = np.zeros( ( cSh[-2], cSh[0]*cSh[-1] + (cSh[0]-1)*gapH ), dtype=np.float32  )
    for curI in range(cSh[0]) :
        collage[ : , curI * (cSh[-1]+gapH) : curI * (cSh[-1]+gapH) + cSh[-1]] = colImgs[curI,...]
    writer.add_scalars("Probs of ref images",
                       {'Ref':probs[0]
                       ,'Gen':probs[2] - probs[0]
                       ,'Pre':probs[1] - probs[0]
                       }, iter )
    writer.add_scalars("Dist of ref images",
                       { 'REC' : dists[0]
                       , 'MSE' : dists[1]
                       , 'L1L' : dists[2]
                       }, iter )
    try :
        addToHDF(TCfg.historyHDF, "data", collage)
    except :
        eprint("Failed to save.")
    return collage, probs, dists


def initialTest() :
    with torch.inference_mode() :
        collage, probs, _ = logStep(iter, not iter)
        print("Probabilities of reference images: "
              f'Ref: {probs[0]:.3e}, '
              f'Gen: {probs[2]:.3e}, '
              f'Pre: {probs[1]:.3e}.')
        generator.eval()
        pre = realModel(generator).preProc(refImages)
        ref_loss_Rec = loss_Rec(refImages[DCfg.gapRng], pre, calculateWeights(refImages))
        ref_loss_MSE = loss_MSE(refImages[DCfg.gapRng], pre)
        ref_loss_L1L = loss_L1L(refImages[DCfg.gapRng], pre)
        print("Distances of reference images: "
              f"REC: {ref_loss_Rec:.3e}, "
              f"MSE: {ref_loss_MSE:.3e}, "
              f"L1L: {ref_loss_L1L:.3e}.")
        if not epoch :
            writer.add_scalars("Dist of ref images",
                                  { 'REC' : ref_loss_Rec
                                  , 'MSE' : ref_loss_MSE
                                  , 'L1L' : ref_loss_L1L
                                  }, 0 )
        plotImage(collage)


def calculateWeights(images) :
    return None


def imagesPreProc(images) :
    return images, None

def imagesPostProc(images, procData=None) :
    return images


noAdv=False


@dataclass
class TrainInfoClass:
    bestRealIndex = 0
    worstRealIndex = 0
    bestFakeIndex = 0
    worstFakeIndex = 0
    bestRealProb = 0
    worstRealProb = 0
    bestFakeProb = 0
    worstFakeProb = 0
    bestRealImage = None
    worstRealImage = None
    bestFakeImage = None
    worstFakeImage = None
    highestDifIndex = 0
    lowestDifIndex = 0
    highestDif = 0
    lowestDif = 0
    highestDifImageOrg = None
    highestDifImageGen = None
    ratReal = 0.0
    ratFake = 0.0
    totalImages = 0
    iterations = 0
    disPerformed = 0
    genPerformed = 0
trainInfo = TrainInfoClass()

withNoGrad = True
def train_step(images):
    global trainDis, trainGen, eDinfo, noAdv
    trainInfo.iterations += 1

    nofIm = images.shape[0]
    images = images.squeeze(1).to(TCfg.device)
    imWeights = calculateWeights(images)

    labelsTrue = torch.full((nofIm, 1),  1 - TCfg.labelSmoothFac,
                        dtype=torch.float, device=TCfg.device)
    labelsFalse = torch.full((nofIm, 1),  TCfg.labelSmoothFac,
                        dtype=torch.float, device=TCfg.device)

    D_loss = torch.zeros(1)
    G_loss = torch.zeros(1)
    GA_loss = torch.zeros(1)
    GD_loss = torch.zeros(1)
    pred_preD = torch.zeros(1)
    pred_realD = torch.zeros(1)
    pred_fakeD = torch.zeros(1)
    pred_fakeG = torch.zeros(1)
    ratReal = 0
    ratFake = 0
    procImages, procReverseData = imagesPreProc(images)

    if not noAdv :

        # calculate predictions of prefilled images - purely for metrics purposes
        discriminator.eval()
        generator.eval()
        fakeImagesD = procImages.clone().detach()
        fakeImagesD.requires_grad=False
        with torch.no_grad() :
            fakeImagesD[DCfg.gapRng] = generator.preProc(procImages)
        pred_preD = discriminator(fakeImagesD)

        # train discriminator
        generator.eval()
        discriminator.train()
        optimizer_D.zero_grad()
        with torch.no_grad() :
            fakeImagesD[()] = generator.generateImages(procImages)
        pred_realD = discriminator(procImages)
        pred_fakeD = discriminator(fakeImagesD)
        generator.preProc(images)
        pred_both = torch.cat((pred_realD, pred_fakeD), dim=0)
        labels = torch.cat( (labelsTrue, labelsFalse), dim=0).to(TCfg.device)
        D_loss = loss_Adv(labels, pred_both,
                            None if imWeights is None else torch.cat( (imWeights, imWeights) )  )
        D_loss.backward()
        optimizer_D.step()
        ratReal = torch.count_nonzero(pred_realD > 0.5)/nofIm
        ratFake = torch.count_nonzero(pred_fakeD > 0.5)/nofIm

    # train generator
    discriminator.eval()
    generator.train()
    optimizer_G.zero_grad()
    fakeImagesG = None
    if isinstance(generator, nn.DataParallel) :
        fakeImagesG = procImages.clone()
        fakeImagesG[DCfg.gapRng] = generator((procImages, None))
    else :
        fakeImagesG = generator.generateImages(procImages)
    if noAdv :
        G_loss = loss_Rec(procImages[DCfg.gapRng], fakeImagesG[DCfg.gapRng],
                 imWeights)
        GD_loss = GA_loss = G_loss
    else :
        pred_fakeG=None
        if withNoGrad :
            with torch.no_grad() :
                pred_fakeG = discriminator(fakeImagesG)
        else :
            pred_fakeG = discriminator(fakeImagesG)
        GA_loss, GD_loss = loss_Gen(labelsTrue, pred_fakeG,
                                    procImages[DCfg.gapRng], fakeImagesG[DCfg.gapRng],
                                    imWeights)
        G_loss = GA_loss + lossDifCoef * GD_loss
        ratFake = torch.count_nonzero(pred_fakeG > 0.5)/nofIm
    G_loss.backward()
    optimizer_G.step()

    MSE_loss = L1L_loss = None
    with torch.no_grad() :

        deprocFakeImages = imagesPostProc(fakeImagesG, procReverseData)
        MSE_loss = loss_MSE(images[DCfg.gapRng], deprocFakeImages[DCfg.gapRng])
        L1L_loss = loss_L1L(images[DCfg.gapRng], deprocFakeImages[DCfg.gapRng])

        idx = random.randint(0, nofIm-1) if noAdv else pred_realD.argmax()
        trainInfo.bestRealImage = deprocFakeImages[idx,...].clone().detach() if noAdv \
                                  else images[idx,...].clone().detach()
        trainInfo.bestRealProb = 0 if noAdv else pred_realD[idx].item()
        trainInfo.bestRealIndex = idx

        idx = random.randint(0, nofIm-1) if noAdv else pred_realD.argmin()
        trainInfo.worstRealImage = deprocFakeImages[idx,...].clone().detach() if noAdv \
                                   else images[idx,...].clone().detach()
        trainInfo.worstRealProb = 0 if noAdv else pred_realD[idx].item()
        trainInfo.worstRealIndex = idx

        idx = random.randint(0, nofIm-1) if noAdv else pred_fakeG.argmax()
        trainInfo.bestFakeImage = deprocFakeImages[idx,...].clone().detach()
        trainInfo.bestFakeProb = 0 if noAdv else pred_fakeG[idx].item()
        trainInfo.bestFakeIndex = idx

        idx = random.randint(0, nofIm-1) if noAdv else pred_fakeG.argmin()
        trainInfo.worstFakeImage = deprocFakeImages[idx,...].clone().detach()
        trainInfo.worstFakeProb = 0 if noAdv else pred_fakeG[idx].item()
        trainInfo.worstFakeIndex = idx

        trainInfo.highestDifIndex = eDinfo[0]
        trainInfo.highestDif = eDinfo[1]
        trainInfo.highestDifImageOrg = images[trainInfo.highestDifIndex,...].clone().detach()
        trainInfo.highestDifImageGen = deprocFakeImages[trainInfo.highestDifIndex,...].clone().detach()
        trainInfo.lowestDifIndex = eDinfo[2]
        trainInfo.lowestDif = eDinfo[3]

        trainInfo.ratReal += ratReal * nofIm
        trainInfo.ratFake += ratFake * nofIm
        trainInfo.totalImages += nofIm

    return D_loss, GA_loss, GD_loss, MSE_loss, L1L_loss, \
           pred_realD.mean(), pred_preD.mean(), pred_fakeD.mean()


epoch=initToNone('epoch')
iter = initToNone('iter')
minGEpoch = initToNone('minGEpoch')
minGdLoss = initToNone('minGdLoss')
prepGdLoss = initToNone('prepGdLoss')


def beforeEachEpoch(epoch) :
    return


def afterEachEpoch(epoch) :
    return


dataLoader=None
testLoader=None

def train(savedCheckPoint):
    global epoch, minGdLoss, minGEpoch, prepGdLoss, iter
    lastGdLoss = minGdLoss

    discriminator.to(TCfg.device)
    generator.to(TCfg.device)
    lastUpdateTime = time.time()

    while TCfg.nofEpochs is None or epoch <= TCfg.nofEpochs :
        epoch += 1
        beforeEachEpoch(epoch)
        generator.train()
        discriminator.train()
        lossDacc = 0
        lossGAacc = 0
        lossGDacc = 0
        lossMSEacc = 0
        lossL1Lacc = 0
        predRealAcc = 0
        predPreAcc = 0
        predFakeAcc = 0
        totalIm = 0

        for it , data in tqdm.tqdm(enumerate(dataLoader), total=int(len(dataLoader))):
            iter += 1

            images = data[0].to(TCfg.device)
            nofIm = images.shape[0]
            totalIm += nofIm
            D_loss, GA_loss, GD_loss, MSE_loss, L1L_loss, \
            predReal, predPre, predFake \
                = train_step(images)
            lossDacc += D_loss.item() * nofIm
            lossGAacc += GA_loss.item() * nofIm
            lossGDacc += GD_loss.item() * nofIm
            lossMSEacc += MSE_loss.item() * nofIm
            lossL1Lacc += L1L_loss.item() * nofIm
            predRealAcc += predReal.item() * nofIm
            predPreAcc  += predPre.item() * nofIm
            predFakeAcc += predFake.item() * nofIm

            #if True:
            #if False :
            #if not it or it > len(dataloader)-2 or time.time() - lastUpdateTime > 60 :
            if time.time() - lastUpdateTime > 60 :

                lastUpdateTime = time.time()
                _,_,_ =  logStep(iter)
                collageR, probsR, _ = generateDiffImages(refImages[[0],...], layout=0)
                showMe = np.zeros( (2*DCfg.sinoSh[1] + DCfg.gapW ,
                                    5*DCfg.sinoSh[0] + 4*DCfg.gapW), dtype=np.float32  )
                for clmn in range (5) : # mark gaps
                    showMe[ DCfg.sinoSh[0] : DCfg.sinoSh[0] + DCfg.gapW ,
                            clmn*(DCfg.sinoSh[1]+DCfg.gapW) + 2*DCfg.gapW : clmn*(DCfg.sinoSh[1]+DCfg.gapW) + 3*DCfg.gapW ] = -1
                def addImage(clmn, row, img, stretch=True) :
                    imgToAdd = img.clone().detach().squeeze()
                    if stretch :
                        minv = imgToAdd.min()
                        ampl = imgToAdd.max() - minv
                        imgToAdd[()] = 2 * ( imgToAdd - minv ) / ampl - 1  if ampl!=0.0 else 0
                    showMe[ row * ( DCfg.sinoSh[1]+DCfg.gapW) : (row+1) * DCfg.sinoSh[1] + row*DCfg.gapW ,
                            clmn * ( DCfg.sinoSh[0]+DCfg.gapW) : (clmn+1) * DCfg.sinoSh[0] + clmn*DCfg.gapW ] = \
                        imgToAdd.cpu().numpy()
                addImage(0,0,trainInfo.bestRealImage)
                addImage(0,1,trainInfo.worstRealImage)
                addImage(1,0,trainInfo.bestFakeImage)
                addImage(1,1,trainInfo.worstFakeImage)
                addImage(2,0,trainInfo.highestDifImageGen)
                addImage(2,1,trainInfo.highestDifImageOrg)
                addImage(3,0,collageR[0,2])
                addImage(3,1,collageR[0,0])
                addImage(4,0,collageR[0,1])
                addImage(4,1,collageR[0,3], stretch=False)
                writer.add_scalars("Losses per iter",
                                   {'Dis': D_loss
                                   ,'Adv': GA_loss
                                   ,'Gen': GA_loss + lossDifCoef * GD_loss
                                   }, iter )
                writer.add_scalars("Distances per iter",
                                   {'MSE': MSE_loss
                                   ,'L1L': L1L_loss
                                   ,'REC': GD_loss
                                   }, iter )
                writer.add_scalars("Probs per iter",
                                   {'Ref':predReal
                                   ,'Gen':predFake
                                   ,'Pre':predPre
                                   }, epoch )


                IPython.display.clear_output(wait=True)
                print(f"Epoch: {epoch} ({minGEpoch}). Losses: " +
                      ( f" L1L: {L1L_loss.item():.3e} " if noAdv \
                          else f" Dis: {D_loss.item():.3f} ({trainInfo.ratReal/trainInfo.totalImages:.3f})," ) +
                      ( f" MSE: {MSE_loss.item():.3e} " if noAdv \
                          else f" Gen: {GA_loss.item():.3f} ({trainInfo.ratFake/trainInfo.totalImages:.3f})," ) +
                      f" Rec: {lastGdLoss:.3e} ({minGdLoss:.3e} / {prepGdLoss:.3e})."
                      )
                trainInfo.iterations = 0
                trainInfo.totalImages = 0
                trainInfo.ratReal = 0
                trainInfo.ratFake = 0
                trainInfo.genPerformed = 0
                trainInfo.disPerformed = 0

                print (f"TT: {trainInfo.bestRealProb:.4e} ({data[1][trainInfo.bestRealIndex]},{data[2][trainInfo.bestRealIndex]}),  "
                       f"FT: {trainInfo.bestFakeProb:.4e} ({data[1][trainInfo.bestFakeIndex]},{data[2][trainInfo.bestFakeIndex]}),  "
                       f"HD: {trainInfo.highestDif:.3e} ({data[1][trainInfo.highestDifIndex]},{data[2][trainInfo.highestDifIndex]}),  "
                       f"GP: {probsR[0,2].item():.5f}, {probsR[0,1].item():.5f} " )
                print (f"TF: {trainInfo.worstRealProb:.4e} ({data[1][trainInfo.worstRealIndex]},{data[2][trainInfo.worstRealIndex]}),  "
                       f"FF: {trainInfo.worstFakeProb:.4e} ({data[1][trainInfo.worstFakeIndex]},{data[2][trainInfo.worstFakeIndex]}),  "
                       f"LD: {trainInfo.lowestDif:.3e} ({data[1][trainInfo.lowestDifIndex]},{data[2][trainInfo.lowestDifIndex]}),  "
                       f"R : {probsR[0,0].item():.5f}." )
                plotImage(showMe)

        Rec_test, MSE_test, L1L_test = summarizeSet(testLoader, False)
        writer.add_scalars("Test per epoch",
                           {'MSE': MSE_test
                           ,'L1L': L1L_test
                           ,'REC': Rec_test
                           }, epoch )

        lossDacc /= totalIm
        lossGAacc /= totalIm
        lossGDacc /= totalIm
        lossMSEacc /= totalIm
        lossL1Lacc /= totalIm
        predRealAcc /= totalIm
        predPreAcc /= totalIm
        predFakeAcc /= totalIm

        writer.add_scalars("Losses per epoch",
                           {'Dis': lossDacc
                           ,'Adv': lossGAacc
                           ,'Gen': lossGAacc + lossDifCoef * lossGDacc
                           }, epoch )
        writer.add_scalars("Distances per epoch",
                           {'MSE': lossMSEacc
                           ,'L1L': lossL1Lacc
                           ,'REC': lossGDacc
                           }, epoch )
        writer.add_scalars("Probs per epoch",
                           {'Ref': predRealAcc
                           ,'Gen': predFakeAcc
                           ,'Pre': predPreAcc
                           }, epoch )
        lastGdLoss = lossGDacc
        if minGdLoss < 0.0 or lossGDacc < minGdLoss  :
            minGdLoss = lossGDacc
            minGEpoch = epoch
            saveCheckPoint(savedCheckPoint+"_B.pth",
                           epoch, iter, minGEpoch, minGdLoss,
                           generator, discriminator,
                           optimizer_G, optimizer_D)
            os.system(f"cp {savedCheckPoint}.pth {savedCheckPoint}_BB.pth") # BB: before best
            os.system(f"cp {savedCheckPoint}_B.pth {savedCheckPoint}.pth") # B: best
            saveModels()
        else :
            saveCheckPoint(savedCheckPoint+".pth",
                           epoch, iter, minGEpoch, minGdLoss,
                           generator, discriminator,
                           optimizer_G, optimizer_D)

        afterEachEpoch(epoch)


def testMe(trainSet, nofIm=1) :
    testSet = [ trainSet.__getitem__() for _ in range(nofIm) ]
    testImages = torch.stack( [ testItem[0] for testItem in testSet ] ).to(TCfg.device)
    colImgs, probs, dists = generateDiffImages(testImages, layout=4)
    for im in range(nofIm) :
        testItem = testSet[im]
        print(f"Index: ({testItem[1], testItem[2]})")
        print(f"Probabilities. Org: {probs[im,0]:.3e},  Gen: {probs[im,2]:.3e},  Pre: {probs[im,1]:.3e}.")
        print(f"Distances. Rec: {dists[im,0]:.4e},  MSE: {dists[im,1]:.4e},  L1L: {dists[im,2]:.4e}.")
        plotImage(colImgs[im].squeeze().cpu())


def freeGPUmem() :
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


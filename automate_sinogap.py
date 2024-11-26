#!/usr/bin/env python3

import math
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
import itertools
import random
import time
import os
import tqdm

import sinogap_module as sg

sg.plt.rcParams['figure.dpi']=223


def my_train_step(images):
    global trainDis, trainGen, eDinfo
    sg.trainInfo.iterations += 1

    nofIm = images.shape[0]
    images = images.squeeze(1).to(sg.TCfg.device)
    imWeights = sg.calculateWeights(images)

    D_loss = None
    G_loss = None
    #GA_loss = None
    #GD_loss = None
    ratReal = 0
    ratFake = 0
    procImages, procReverseData = sg.imagesPreProc(images)

    # calculate predictions of prefilled images - purely for metrics purposes
    #sg.discriminator.eval()
    #sg.generator.eval()
    #fakeImagesD = procImages.clone().detach()
    #fakeImagesD.requires_grad=False
    #with torch.no_grad() :
    #    fakeImagesD[sg.DCfg.gapRng] = sg.generator.preProc(procImages)
    #pred_preD = sg.discriminator(fakeImagesD)

    #try :
    #    generator.eval()
    #    discriminator.train()
    #    optimizer_D.zero_grad()
    #    with torch.no_grad() :
    #        fakeImagesD[()] = generator.generateImages(procImages)
    #    pred_realD = discriminator(procImages)
    #    pred_fakeD = discriminator(fakeImagesD)
    #    generator.preProc(images)
    #
    #    pred_both = torch.cat((pred_realD, pred_fakeD), dim=0)
    #    labels = torch.cat( (labelsTrue, labelsFalse), dim=0).to(TCfg.device)
    #    D_loss = loss_Adv(labels, pred_both,
    #                      None if imWeights is None else torch.cat( (imWeights, imWeights) )  )
    #    D_loss.backward()
    #    optimizer_D.step()
    #except :
    #    optimizer_D.zero_grad()
    #    del fakeImagesD
    #    del pred_realD
    #    del pred_fakeD
    #    del pred_both
    #    del D_loss
    #    raise
    #ratReal = torch.count_nonzero(pred_realD > 0.5)/nofIm
    #ratFake = torch.count_nonzero(pred_fakeD > 0.5)/nofIm

    try :
        #discriminator.eval()
        sg.generator.train()
        sg.optimizer_G.zero_grad()
        fakeImagesG = sg.generator.generateImages(procImages)
        #pred_fakeG=None
        #if sg.withNoGrad :
        #    with torch.no_grad() :
        #        pred_fakeG = discriminator(fakeImagesG)
        #else :
        #    pred_fakeG = discriminator(fakeImagesG)
        #GA_loss, GD_loss = loss_Gen(labelsTrue, pred_fakeG,
        #                            procImages[DCfg.gapRng], fakeImagesG[DCfg.gapRng],
        #                            imWeights)
        #G_loss = GA_loss + lossDifCoef * GD_loss
        G_loss = sg.loss_Rec(procImages[sg.DCfg.gapRng], fakeImagesG[sg.DCfg.gapRng],
                             imWeights)
        G_loss.backward()
        sg.optimizer_G.step()
    except :
        sg.optimizer_G.zero_grad()
        del fakeImagesG
        #del pred_fakeG
        del G_loss
        #del GA_loss
        #del GD_loss
        raise
    #ratFake = torch.count_nonzero(pred_fakeG > 0.5)/nofIm

    MSE_loss = L1L_loss = None
    with torch.no_grad() :

        deprocFakeImages = sg.imagesPostProc(fakeImagesG, procReverseData)
        MSE_loss = sg.loss_MSE(images[sg.DCfg.gapRng], deprocFakeImages[sg.DCfg.gapRng])
        L1L_loss = sg.loss_L1L(images[sg.DCfg.gapRng], deprocFakeImages[sg.DCfg.gapRng])

        idx = random.randint(0, nofIm-1) # pred_realD.argmax()
        sg.trainInfo.bestRealImage = deprocFakeImages[idx,...].clone().detach()
        sg.trainInfo.bestRealProb = sg.MSE(images[idx,*sg.DCfg.gapRng],
                                           deprocFakeImages[idx,*sg.DCfg.gapRng]).mean().item()
        sg.trainInfo.bestRealIndex = idx

        idx = random.randint(0, nofIm-1) #pred_realD.argmin()
        sg.trainInfo.worstRealImage = deprocFakeImages[idx,...].clone().detach()
        sg.trainInfo.worstRealProb = sg.MSE(images[idx,*sg.DCfg.gapRng],
                                            deprocFakeImages[idx,*sg.DCfg.gapRng]).mean().item()
        sg.trainInfo.worstRealIndex = idx

        idx = random.randint(0, nofIm-1) #pred_fakeG.argmax()
        sg.trainInfo.bestFakeImage = deprocFakeImages[idx,...].clone().detach()
        sg.trainInfo.bestFakeProb = sg.MSE(images[idx,*sg.DCfg.gapRng],
                                           deprocFakeImages[idx,*sg.DCfg.gapRng]).mean().item()
        sg.trainInfo.bestFakeIndex = idx

        idx = random.randint(0, nofIm-1) #pred_fakeG.argmin()
        sg.trainInfo.worstFakeImage = deprocFakeImages[idx,...].clone().detach()
        sg.trainInfo.worstFakeProb = sg.MSE(images[idx,*sg.DCfg.gapRng],
                                           deprocFakeImages[idx,*sg.DCfg.gapRng]).mean().item()
        sg.trainInfo.worstFakeIndex = idx

        sg.trainInfo.highestDifIndex = sg.eDinfo[0]
        sg.trainInfo.highestDif = sg.eDinfo[1]
        sg.trainInfo.highestDifImageOrg = images[sg.trainInfo.highestDifIndex,...].clone().detach()
        sg.trainInfo.highestDifImageGen = deprocFakeImages[sg.trainInfo.highestDifIndex,...].clone().detach()
        sg.trainInfo.lowestDifIndex = sg.eDinfo[2]
        sg.trainInfo.lowestDif = sg.eDinfo[3]

        sg.trainInfo.ratReal += ratReal * nofIm
        sg.trainInfo.ratFake += ratFake * nofIm
        sg.trainInfo.totalImages += nofIm

    return G_loss, G_loss, G_loss, MSE_loss, L1L_loss, \
           torch.zeros(1),  torch.zeros(1),  torch.zeros(1)

sg.train_step = my_train_step


def my_train(savedCheckPoint):
    #global sg.epoch, sg.minGdLoss, sg.minGEpoch, sg.prepGdLoss, sg.iter


    sg.discriminator.to(sg.TCfg.device)
    sg.generator.to(sg.TCfg.device)
    lastUpdateTime = time.time()
    startTime = time.time()

    pbar = tqdm.tqdm(total=sg.TCfg.nofEpochs)
    #while sg.epoch is None or sg.epoch <= sg.TCfg.nofEpochs :
    while time.time() - startTime < 10000 :
        sg.epoch += 1
        sg.beforeEachEpoch(sg.epoch)
        sg.generator.train()
        sg.discriminator.train()
        lossDacc = 0
        lossGAacc = 0
        lossGDacc = 0
        lossMSEacc = 0
        lossL1Lacc = 0
        predRealAcc = 0
        predPreAcc = 0
        predFakeAcc = 0
        totalIm = 0


        for data in sg.dataLoader :
            sg.iter += 1

            images = data[0].to(sg.TCfg.device)
            nofIm = images.shape[0]
            totalIm += nofIm
            D_loss, GA_loss, GD_loss, MSE_loss, L1L_loss, \
            predReal, predPre, predFake \
                = sg.train_step(images)
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
                _,_,_ =  sg.logStep(sg.iter)
                collageR, probsR, _ = sg.generateDiffImages(sg.refImages[[0],...], layout=0)
                showMe = np.zeros( (2*sg.DCfg.sinoSh[1] + sg.DCfg.gapW ,
                                    5*sg.DCfg.sinoSh[0] + 4*sg.DCfg.gapW), dtype=np.float32  )
                for clmn in range (5) : # mark gaps
                    showMe[ sg.DCfg.sinoSh[0] : sg.DCfg.sinoSh[0] + sg.DCfg.gapW ,
                            clmn*(sg.DCfg.sinoSh[1]+sg.DCfg.gapW) + 2*sg.DCfg.gapW : clmn*(sg.DCfg.sinoSh[1]+sg.DCfg.gapW) + 3*sg.DCfg.gapW ] = -1
                def addImage(clmn, row, img, stretch=True) :
                    imgToAdd = img.clone().detach().squeeze()
                    if stretch :
                        minv = imgToAdd.min()
                        ampl = imgToAdd.max() - minv
                        imgToAdd[()] = 2 * ( imgToAdd - minv ) / ampl - 1  if ampl!=0.0 else 0
                    showMe[ row * ( sg.DCfg.sinoSh[1]+sg.DCfg.gapW) : (row+1) * sg.DCfg.sinoSh[1] + row*sg.DCfg.gapW ,
                            clmn * ( sg.DCfg.sinoSh[0]+sg.DCfg.gapW) : (clmn+1) * sg.DCfg.sinoSh[0] + clmn*sg.DCfg.gapW ] = \
                        imgToAdd.cpu().numpy()
                addImage(0,0,sg.trainInfo.bestRealImage)
                addImage(0,1,sg.trainInfo.worstRealImage)
                addImage(1,0,sg.trainInfo.bestFakeImage)
                addImage(1,1,sg.trainInfo.worstFakeImage)
                addImage(2,0,sg.trainInfo.highestDifImageGen)
                addImage(2,1,sg.trainInfo.highestDifImageOrg)
                addImage(3,0,collageR[0,2])
                addImage(3,1,collageR[0,0])
                addImage(4,0,collageR[0,1])
                addImage(4,1,collageR[0,3], stretch=False)
                sg.writer.add_scalars("Losses per iter",
                                   {'Dis': D_loss
                                   ,'Adv': GA_loss
                                   ,'Gen': GA_loss + sg.lossDifCoef * GD_loss
                                   }, sg.iter )
                sg.writer.add_scalars("Distances per iter",
                                   {'MSE': MSE_loss
                                   ,'L1L': L1L_loss
                                   ,'REC': GD_loss
                                   }, sg.iter )
                sg.writer.add_scalars("Probs per iter",
                                   {'Ref':predReal
                                   ,'Gen':predFake
                                   ,'Pre':predPre
                                   }, sg.epoch )


                #IPython.display.clear_output(wait=True)
                #print(f"Epoch: {epoch} ({minGEpoch}). Losses: "
                #      f" Dis: {D_loss:.3f} "
                #      f"({sg.trainInfo.ratReal/sg.trainInfo.totalImages:.3f}),"
                #      f" Gen: {GA_loss:.3f} "
                #      f"({sg.trainInfo.ratFake/sg.trainInfo.totalImages:.3f}),"
                #      f" Rec: {lastGdLoss:.3e} ({minGdLoss:.3e} / {prepGdLoss:.3e})."
                #      )
                sg.trainInfo.iterations = 0
                sg.trainInfo.totalImages = 0
                sg.trainInfo.ratReal = 0
                sg.trainInfo.ratFake = 0
                sg.trainInfo.genPerformed = 0
                sg.trainInfo.disPerformed = 0

                #print (f"TT: {sg.trainInfo.bestRealProb:.4e} ({data[1][sg.trainInfo.bestRealIndex]},{data[2][sg.trainInfo.bestRealIndex]}),  "
                #       f"FT: {sg.trainInfo.bestFakeProb:.4e} ({data[1][sg.trainInfo.bestFakeIndex]},{data[2][sg.trainInfo.bestFakeIndex]}),  "
                #       f"HD: {sg.trainInfo.highestDif:.3e} ({data[1][sg.trainInfo.highestDifIndex]},{data[2][sg.trainInfo.highestDifIndex]}),  "
                #       f"GP: {probsR[0,2].item():.5f}, {probsR[0,1].item():.5f} " )
                #print (f"TF: {sg.trainInfo.worstRealProb:.4e} ({data[1][sg.trainInfo.worstRealIndex]},{data[2][sg.trainInfo.worstRealIndex]}),  "
                #       f"FF: {sg.trainInfo.worstFakeProb:.4e} ({data[1][sg.trainInfo.worstFakeIndex]},{data[2][sg.trainInfo.worstFakeIndex]}),  "
                #       f"LD: {sg.trainInfo.lowestDif:.3e} ({data[1][sg.trainInfo.lowestDifIndex]},{data[2][sg.trainInfo.lowestDifIndex]}),  "
                #       f"R : {probsR[0,0].item():.5f}." )
                #plotImage(showMe)

        lossDacc /= totalIm
        lossGAacc /= totalIm
        lossGDacc /= totalIm
        lossMSEacc /= totalIm
        lossL1Lacc /= totalIm
        predRealAcc /= totalIm
        predPreAcc /= totalIm
        predFakeAcc /= totalIm

        sg.writer.add_scalars("Losses per epoch",
                           {'Dis': lossDacc
                           ,'Adv': lossGAacc
                           ,'Gen': lossGAacc + sg.lossDifCoef * lossGDacc
                           }, sg.epoch )
        sg.writer.add_scalars("Distances per epoch",
                           {'MSE': lossMSEacc
                           ,'L1L': lossL1Lacc
                           ,'REC': lossGDacc
                           }, sg.epoch )
        sg.writer.add_scalars("Probs per epoch",
                           {'Ref': predRealAcc
                           ,'Gen': predFakeAcc
                           ,'Pre': predPreAcc
                           }, sg.epoch )
        lastGdLoss = lossGDacc
        if sg.minGdLoss < 0.0 or lossGDacc < sg.minGdLoss  :
            sg.minGdLoss = lossGDacc
            sg.minGEpoch = sg.epoch
            sg.saveCheckPoint(savedCheckPoint+"_BEST.pth",
                           sg.epoch, sg.iter, sg.minGEpoch, sg.minGdLoss,
                           sg.generator, sg.discriminator,
                           sg.optimizer_G, sg.optimizer_D)
            os.system(f"cp {savedCheckPoint}.pth {savedCheckPoint}_BeforeBest.pth")
            os.system(f"cp {savedCheckPoint}_BEST.pth {savedCheckPoint}.pth")
            sg.saveModels()
        else :
            sg.saveCheckPoint(savedCheckPoint+".pth",
                           sg.epoch, sg.iter, sg.minGEpoch, sg.minGdLoss,
                           sg.generator, sg.discriminator,
                           sg.optimizer_G, sg.optimizer_D)

        sg.afterEachEpoch(sg.epoch)
        pbar.update(1)


sg.train = my_train



# %% [markdown]
# ### <font style="color:lightblue">Configs</font>

# %%
sg.set_seed(7)

sg.TCfg = sg.TCfgClass(
     exec = 1
    ,nofEpochs = 50
    ,latentDim = 64
    ,batchSize=128
    ,labelSmoothFac = 0.1 # For Fake labels (or set to 0.0 for no smoothing).
    ,learningRateD = 0.0002
    ,learningRateG = 0.0002
)

sg.DCfg = sg.DCfgClass(8)


# %% [markdown]
# ### <font style="color:lightblue">Raw Read</font>

# %%
trainSet = sg.createTrainSet()
sg.prepGdLoss=0

# %% [markdown]
# ### <font style="color:lightblue">Show</font>

# %%

sg.refImages, sg.refNoises = sg.createReferences(trainSet, 0)
#sg.showMe(trainSet, 0)


# %%


# %% [markdown]
# ## <font style="color:lightblue">Models</font>

# %% [markdown]
# ### <font style="color:lightblue">Generator</font>

# %%



class Generator2(sg.GeneratorTemplate):

    def __init__(self):
        super(Generator2, self).__init__(2)

        #latentChannels = 7
        #self.noise2latent = nn.Sequential(
        #    nn.Linear(sg.TCfg.latentDim, self.sinoSize*latentChannels),
        #    nn.ReLU(),
        #    nn.Unflatten( 1, (latentChannels,) + self.sinoSh )
        #)

        baseChannels = 64

        def encblock(chIn, chOut, kernel, stride=1, norm=True) :
            layers = []
            layers.append(nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True))
            if norm :
                layers.append(nn.BatchNorm2d(chOut))
            layers.append(nn.LeakyReLU(0.2))
            sg.fillWheights(layers)
            return torch.nn.Sequential(*layers)
        self.encoders =  nn.ModuleList([
            encblock(  1,                baseChannels, 3, norm=False),
            encblock(  baseChannels,     baseChannels, 3),
            encblock(  baseChannels,     baseChannels, 3),
            ])

        smpl = torch.zeros((1,1,*self.sinoSh))
        for encoder in self.encoders :
            smpl = encoder(smpl)
        encSh = smpl.shape
        linChannels = math.prod(encSh)
        self.fcLink = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        sg.fillWheights(self.fcLink)

        def decblock(chIn, chOut, kernel, stride=1, norm=True) :
            layers = []
            layers.append(nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True))
            if norm :
                layers.append(nn.BatchNorm2d(chOut))
            layers.append(nn.LeakyReLU(0.2))
            sg.fillWheights(layers)
            return torch.nn.Sequential(*layers)
        self.decoders = nn.ModuleList([
            decblock(2*baseChannels, baseChannels, 3),
            decblock(2*baseChannels, baseChannels, 3),
            decblock(2*baseChannels, baseChannels, 3, norm=False),
            ])

        self.lastTouch = nn.Sequential(
            nn.Conv2d(baseChannels+1, 1, 1),
            nn.Tanh(),
        )
        sg.fillWheights(self.lastTouch)


    def forward(self, input):

        images, _ = input
        images, orgDims = sg.unsqeeze4dim(images)
        modelIn = images.clone()
        modelIn[self.gapRng] = self.preProc(images)

        minv = modelIn.min(dim=-1).values.min(dim=-1).values
        ampl = modelIn.max(dim=-1).values.max(dim=-1).values - minv
        minv = minv[:,:,None,None]
        ampl = ampl[:,:,None,None]
        iampl = torch.where(ampl==0, 0, 2/ampl)
        modelIn = ( modelIn - minv ) * iampl - 1 # stretch

        #latent = self.noise2latent(noises)
        #modelIn = torch.cat((modelIn,latent),dim=1).to(sg.TCfg.device)
        dwTrain = [modelIn,]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
        res = self.lastTouch(torch.cat( (upTrain[-1], modelIn), dim=1 ))

        patches = ( 2*res[self.gapRng] + modelIn[:,[0],:, self.gapRngX] + 1 ) * ampl / 2 + minv #destretch
        return sg.squeezeOrg(patches, orgDims)

generator2 = Generator2()
generator2 = sg.load_model(generator2, model_path="saves/gen2.pt" )
generator2 = generator2.to(sg.TCfg.device)
generator2 = generator2.requires_grad_(False)
generator2 = generator2.eval()
sg.lowResGenerators[2] = generator2




class Generator4(sg.GeneratorTemplate):

    def __init__(self):
        super(Generator4, self).__init__(4)

        #latentChannels = 7
        #self.noise2latent = nn.Sequential(
        #    nn.Linear(sg.TCfg.latentDim, self.sinoSize*latentChannels),
        #    nn.ReLU(),
        #    nn.Unflatten( 1, (latentChannels,) + self.sinoSh )
        #)

        baseChannels = 64

        def encblock(chIn, chOut, kernel, stride=1, norm=True, dopadding=False) :
            layers = []
            layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True,
                                     padding='same', padding_mode='reflect') \
                           if stride == 1 and dopadding else \
                           nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True)
                           )
            if norm :
                layers.append(nn.BatchNorm2d(chOut))
            layers.append(nn.LeakyReLU(0.2))
            sg.fillWheights(layers)
            return torch.nn.Sequential(*layers)
        self.encoders =  nn.ModuleList([
            encblock(  1,                  baseChannels, 3, norm=False),
            encblock(  baseChannels,     2*baseChannels, 3, stride=2),
            encblock(  2*baseChannels,   2*baseChannels, 3),
            encblock(  2*baseChannels,   2*baseChannels, 3),
            ])

        smpl = torch.zeros((1,1,*self.sinoSh))
        for encoder in self.encoders :
            smpl = encoder(smpl)
        encSh = smpl.shape
        linChannels = math.prod(encSh)
        self.fcLink = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        sg.fillWheights(self.fcLink)

        def decblock(chIn, chOut, kernel, stride=1, norm=True) :
            layers = []
            layers.append(nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True))
            if norm :
                layers.append(nn.BatchNorm2d(chOut))
            layers.append(nn.LeakyReLU(0.2))
            sg.fillWheights(layers)
            return torch.nn.Sequential(*layers)
        self.decoders = nn.ModuleList([
            decblock(4*baseChannels, 2*baseChannels, 3),
            decblock(4*baseChannels, 2*baseChannels, 3),
            decblock(4*baseChannels,   baseChannels, 4, stride=2),
            decblock(2*baseChannels,   baseChannels, 3, norm=False),
            ])

        self.lastTouch = nn.Sequential(
            nn.Conv2d(baseChannels+1, 1, 1),
            nn.Tanh(),
        )
        sg.fillWheights(self.lastTouch)


    def forward(self, input):

        images, noises = input
        images, orgDims = sg.unsqeeze4dim(images)
        modelIn = images.clone()
        modelIn[self.gapRng] = self.preProc(images)

        #latent = self.noise2latent(noises)
        #modelIn = torch.cat((modelIn,latent),dim=1).to(sg.TCfg.device)
        dwTrain = [modelIn,]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
        res = self.lastTouch(torch.cat( (upTrain[-1], modelIn ), dim=1 ))

        patches = modelIn[self.gapRng] + 2 * res[self.gapRng]
        return sg.squeezeOrg(patches, orgDims)

generator4 = Generator4()
generator4 = sg.load_model(generator4, model_path="saves/gen4.pt" )
generator4 = generator4.to(sg.TCfg.device)
generator4 = generator4.requires_grad_(False)
generator4 = generator4.eval()
sg.lowResGenerators[4] = generator4





class Generator8(sg.GeneratorTemplate):

    def __init__(self):
        super(Generator8, self).__init__(8)

        #latentChannels = 7
        #self.noise2latent = nn.Sequential(
        #    nn.Linear(sg.TCfg.latentDim, self.sinoSize*latentChannels),
        #    nn.ReLU(),
        #    nn.Unflatten( 1, (latentChannels,) + self.sinoSh )
        #)

        baseChannels = 64

        def encblock(chIn, chOut, kernel, stride=1, norm=True, dopadding=False) :
            layers = []
            layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True,
                                     padding='same', padding_mode='reflect') \
                           if stride == 1 and dopadding else \
                           nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True)
                           )
            if norm :
                layers.append(nn.BatchNorm2d(chOut))
            layers.append(nn.LeakyReLU(0.2))
            sg.fillWheights(layers)
            return torch.nn.Sequential(*layers)
        self.encoders =  nn.ModuleList([
            encblock(  1,                  baseChannels, 3, norm=False),
            encblock(  baseChannels,     2*baseChannels, 3, stride=2),
            encblock(  2*baseChannels,   4*baseChannels, 3, stride=2),
            encblock(  4*baseChannels,   4*baseChannels, 3),
            encblock(  4*baseChannels,   4*baseChannels, 3),
            ])

        smpl = torch.zeros((1,1,*self.sinoSh))
        for encoder in self.encoders :
            smpl = encoder(smpl)
        encSh = smpl.shape
        linChannels = math.prod(encSh)
        self.fcLink = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        sg.fillWheights(self.fcLink)

        def decblock(chIn, chOut, kernel, stride=1, norm=True) :
            layers = []
            layers.append(nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True))
            if norm :
                layers.append(nn.BatchNorm2d(chOut))
            layers.append(nn.LeakyReLU(0.2))
            sg.fillWheights(layers)
            return torch.nn.Sequential(*layers)
        self.decoders = nn.ModuleList([
            decblock(8*baseChannels, 4*baseChannels, 3),
            decblock(8*baseChannels, 4*baseChannels, 3),
            decblock(8*baseChannels, 2*baseChannels, 4, stride=2),
            decblock(4*baseChannels,   baseChannels, 4, stride=2),
            decblock(2*baseChannels,   baseChannels, 3, norm=False),
            ])

        self.lastTouch = nn.Sequential(
            nn.Conv2d(baseChannels+1, 1, 1),
            nn.Tanh(),
        )
        sg.fillWheights(self.lastTouch)


    def forward(self, input):

        images, noises = input
        images, orgDims = sg.unsqeeze4dim(images)
        modelIn = images.clone()
        modelIn[self.gapRng] = self.preProc(images)

        #latent = self.noise2latent(noises)
        #modelIn = torch.cat((modelIn,latent),dim=1).to(sg.TCfg.device)
        dwTrain = [modelIn,]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
        res = self.lastTouch(torch.cat( (upTrain[-1], modelIn ), dim=1 ))

        patches = modelIn[self.gapRng] + 2 * res[self.gapRng]
        return sg.squeezeOrg(patches, orgDims)

sg.generator = Generator8()
sg.generator.to(sg.TCfg.device)




# %% [markdown]
# ### <font style="color:lightblue">Discriminator</font>

# %%

class Discriminator(sg.DiscriminatorTemplate):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.param = nn.Parameter(torch.zeros(1))
    def forward(self, images):
        return torch.zeros((images.shape[0],1), device=sg.TCfg.device)

sg.discriminator = Discriminator()
sg.discriminator = sg.discriminator.to(sg.TCfg.device)
#model_summary = summary(sg.discriminator, input_data=sg.refImages[0,...] ).__str__()
#print(model_summary)
#sg.writer.add_graph(sg.discriminator, refImages)



# %% [markdown]
# ### <font style="color:lightblue">Optimizers</font>

# %%
sg.optimizer_G , sg.optimizer_D = sg.createOptimizers()

# %% [markdown]
# ## <font style="color:lightblue">Restore checkpoint</font>

# %%
savedCheckPoint = f"checkPoint_{sg.TCfg.exec}"
sg.epoch, sg.iter, sg.minGEpoch, sg.minGdLoss = sg.restoreCheckpoint()#savedCheckPoint+".pth")
#sg.epoch, sg.iter = 0 , 0
sg.writer = sg.createWriter(sg.TCfg.logDir, True)
#sg.writer.add_graph(sg.generator, ((sg.refImages, sg.refNoises),) )
#sg.writer.add_graph(sg.discriminator, refImages)
#sg.initialTest()

# %% [markdown]
# ## <font style="color:lightblue">Execute</font>

# %%

#for item in itertools.chain( sg.optimizer_D.param_groups, sg.optimizer_G.param_groups ):
#    item['lr'] *= 0.1
sg.dataLoader = sg.createTrainLoader(trainSet)#, num_workers=0)


#torch.autograd.set_detect_anomaly(True)
#Summary. Rec: 6.114e-04, MSE: 6.114e-04, L1L: 9.813e-03.
sg.prepGdLoss = 6.114e-04
if sg.prepGdLoss == 0:
    Rec_diff, MSE_diff, L1L_diff = sg.summarizeSet(sg.dataLoader)
    sg.prepGdLoss = Rec_diff
    sg.writer.add_scalars("Distances per epoch",
                          {'MSE0': MSE_diff
                          ,'L1L0': L1L_diff
                          ,'REC0': Rec_diff
                          }, 0 )

try :
    sg.train(savedCheckPoint)
except :
    del sg.dataLoader
    sg.freeGPUmem()
    1/10 # to release Jupyuter memory in the next step
    raise

# stretch-add,  wide, links, Diff, 3Layers, eval/train, nograd, double patches, double FC, last-in, pure MSE

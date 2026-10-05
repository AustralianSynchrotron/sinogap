import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class DCfgClass:
    gapW : int
    inChannels : int
    baseChannels : int
    otherChannels : int
    brick : bool = field(repr = False)
    sinoSh : tuple = field(repr = True, init = False)
    gapRng : type(np.s_[:]) = field(repr = True, init = False)
    def __post_init__(self):
        self.sinoSh = ( (8 if self.brick else 256) * self.gapW , 7*self.gapW )
        self.gapRng = np.s_[..., self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]



def normalizeImages(images) :
    images, orgDims = unsqeeze4dim(images)
    stds, means = torch.std_mean(images, dim=(-1,-2), keepdim=True)
    stds = stds + 1e-7
    images = (images - means) / stds # normalize per image
    return images, (orgDims, stds, means)


#def calculateNorm(images) :
#    noGapImages = torch.cat( (images[...,:DCfg.gapRngX.start], images[...,DCfg.gapRngX.stop:]), dim=-1)
#    toRet = torch.std_mean( noGapImages, dim=(-1,-2), keepdim=True )
#    return toRet[0], toRet[1]



def reNormalizeImages(images, norms, stdOnly=False) :
    images = images * norms[1][:,[0],...].to(images.device)
    if not stdOnly :
        images = images + norms[2][:,[0],...].to(images.device) # renormalise
    images = squeezeOrg(images, norms[0])
    return images


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


def stripe2bricks(stripes, ratio=32) :
    width = stripes.shape[-1]
    hight = stripes.shape[-2] // ratio
    channels = stripes.shape[1]
    bricks = stripes.unfold(-2,hight,hight//2).permute(0,2,1,4,3).reshape(-1,channels,hight,width)
    return  bricks


brickMasks = {}
def bricks2stripe(bricks, ratio=32) :
    global brickMasks
    nofIm = bricks.shape[0] // (2*ratio-1)
    channels = bricks.shape[1]
    width = bricks.shape[-1]
    hight = bricks.shape[-2]
    if hight < 2 :
        raise Exception(f"Cant converts bricks {bricks.shape}: impossible bricks height {hight}.")
    if hight not in brickMasks :
        halfLine = [ i + 0.5 for i in range(hight//2)]
        halfLine = torch.tensor(halfLine, dtype=torch.float32)
        halfLine /= hight//2
        line = torch.cat( (halfLine, halfLine.flip(0)), dim=0)
        brickMasks[hight] = line.view(-1,1)
    myMask = brickMasks[hight].repeat(1,width).unsqueeze(0).unsqueeze(0).to(bricks.device)
    bricks = myMask * bricks
    stripesOrg = bricks.view(nofIm,-1,channels,hight,width)[:, ::2,...].transpose(1,2).reshape((nofIm,channels,-1,width))
    stripesAux = bricks.view(nofIm,-1,channels,hight,width)[:,1::2,...].transpose(1,2).reshape((nofIm,channels,-1,width))
    edge = hight//2
    stripes = torch.cat([ stripesOrg[:,:,:edge,:] / myMask[:,:,:edge,:],
                          stripesOrg[:,:,edge:-edge,:] + stripesAux,
                          stripesOrg[:,:,-edge:,:] / myMask[:,:,-edge:,:]
                        ],
                        dim=-2)
    return stripes


def firstDevice(model):
    return next(model.parameters()).device





mainFloors = {
               2  : [
                       ( 1, 2, 3, 2, 1),
                       ( 2, 2, 3, (2,1), (1,0)),
                       ( 4, 2, 3, (2,1), (1,0)),
                    ],
               4  : [
                       ( 1, 2, 3, 2, 1),
                       ( 2, 2, 3, 2, 1),
                       ( 4, 2, 3, (2,1), (1,0)),
                       ( 8, 2, 3, (2,1), (1,0)),
                    ],
               8  : [
                       ( 1, 2, 3, 2, 1),
                       ( 2, 2, 3, 2, 1),
                       ( 4, 2, 3, 2, 1),
                       ( 8, 2, 3, (2,1), (1,0)),
                       (16, 2, 3, (2,1), (1,0)),
                    ],
               16 : [
                       ( 1, 2, 3, 2, 1),
                       ( 2, 2, 3, 2, 1),
                       ( 4, 2, 3, 2, 1),
                       ( 8, 2, 3, 2, 1),
                       (16, 2, 3, (2,1), (1,0)),
                       (32, 2, 3, (2,1), (1,0)),
                    ],
             }


deepFloors = [
    (1,   1/2, 3    , 1, (1,0)),
    (1/2, 1/2, (3,1), 1, (1,0)),
    (1/4, 1/2, (3,1), 1, (1,0)),
    (1/8, 1/2, (3,1), 1, (1,0)),
]



class SubTemplate(nn.Module):

    def __init__(self, gapW, brick, inChannels, baseChannels, otherChannels, floors, outerKernel=3, norms=True):
        super().__init__()
        self.cfg = DCfgClass(gapW, inChannels, baseChannels, otherChannels, brick)
        self.entrance = None if inChannels == baseChannels else \
            self.encblock(inChannels, baseChannels, stride=1, norm=False,
                          kernel=outerKernel, padding=(outerKernel-1)//2 )
        self.encoders = self.createEncoders(floors, norms=norms)


    def encblock(self, chIn, chOut, kernel, stride=1, norm=True, padding=1) :
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias = not norm,
                                padding=padding, padding_mode='reflect')  )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        return torch.nn.Sequential(*layers)


    def encFloor(self, chIn, mult, kernel, stride=1, norm=True, padding=1) :
        firstPadding = (kernel[0]//2, kernel[1]//2) if isinstance(kernel, tuple) else kernel//2
        block1 = self.encblock( int(chIn*(self.cfg.baseChannels+self.cfg.otherChannels)),
                               int(chIn*self.cfg.baseChannels),
                               kernel, stride=1, norm=norm, padding=firstPadding)
        block2 = self.encblock( int(chIn*(self.cfg.baseChannels+self.cfg.otherChannels)),
                               int(chIn*self.cfg.baseChannels*mult),
                               kernel, stride=stride, norm=norm, padding=padding)
        return (block1, block2)


    def createEncoders(self, floors, norms=True) :
        encoders = nn.ModuleList([])
        for floor in floors:
            encoders.extend( self.encFloor(floor[0], mult=floor[1], kernel=floor[2], stride=floor[3], padding=floor[4], norm=norms) )
        return encoders


    def postEncoderShape(self, encoders=None, inShape=None) :
        if encoders is None :
            encoders = self.encoders
        if inShape is None :
            inShape = (1, 1, *self.cfg.sinoSh)
        smpl = torch.zeros(inShape)
        for encoder in encoders :
            smpl = torch.zeros((1, encoder[0].in_channels, *smpl.shape[2:]))
            smpl = encoder(smpl)
        return smpl.shape


    def forward(self, images):
        raise Exception("this is not for direct use")



class SubGeneratorTemplate(SubTemplate):

    def __init__(self, gapW, brick, inChannels, baseChannels, otherChannels, floors, outerKernel=3, noiseChannels=0):
        super().__init__(gapW, brick, inChannels+noiseChannels, baseChannels, otherChannels, floors, outerKernel=outerKernel)
        self.link = None
        self.amplitude = 4 # used to be a parameter
        self.decoders = self.createDecoders(floors)
        self.lastTouch = None if inChannels == baseChannels else self.createLastTouch()
        self.noiseInjector = self.createLatentGenerator(noiseChannels) if noiseChannels else None


    def decblock(self, chIn, chOut, kernel, stride=1, norm=True, padding=1, outputPadding=None) :
        if outputPadding is None :
            if isinstance(stride, int) :
                outputPadding = stride - 1
            else :
                outputPadding = tuple( strd - 1 for strd in stride )
        layers = []
        layers.append( nn.ConvTranspose2d(chIn, chOut, kernel, stride=stride, bias = not norm,
                                          padding=padding, padding_mode='zeros', output_padding=outputPadding) )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        return torch.nn.Sequential(*layers)


    def decFloor(self, chOut, reduce, kernel, stride=1, norm=True, padding=1) :
        block1 = self.decblock( int(reduce*2*chOut*(self.cfg.baseChannels + self.cfg.otherChannels)),
                                int(chOut*self.cfg.baseChannels),
                                kernel, stride=stride, norm=norm, padding=padding)
        secondPadding = (kernel[0]//2, kernel[1]//2) if isinstance(kernel, tuple) else kernel//2
        block2 = self.decblock( int(2*chOut*(self.cfg.baseChannels + self.cfg.otherChannels)),
                                int(chOut*self.cfg.baseChannels),
                                kernel, stride=1, norm=norm, padding=secondPadding)
        return (block1, block2)


    def createDecoders(self, floors) :
        decoders = nn.ModuleList([])
        for floor in reversed(floors):
            decoders.extend( self.decFloor(floor[0], reduce=floor[1], kernel=floor[2], stride=floor[3], padding=floor[4]) )
        return decoders


    def createLastTouch(self) :
        toRet = nn.Sequential(
            nn.Conv2d(self.cfg.baseChannels+self.cfg.otherChannels+self.cfg.inChannels, 1, 1),
            nn.Tanh(),
        )
        return toRet


    def createLatentGenerator(self, outChannels, decoders=None) :
        latInSh = self.postEncoderShape()[-2:]
        latInSh = (1,self.cfg.baseChannels,*latInSh)
        latInChannels = math.prod(latInSh)
        toRet = nn.Sequential(
            nn.Linear(latInChannels, latInChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, latInSh[1:]),
        )
        if decoders is None :
            decoders = self.decoders
        for decoder in decoders :
            conv = decoder[0]
            toRet.append( self.decblock(self.cfg.baseChannels, self.cfg.baseChannels,
                                        kernel=conv.kernel_size,
                                        stride=conv.stride,
                                        norm=False,
                                        padding=conv.padding,
                                        outputPadding=conv.output_padding
                                        ) )
        toRet.append(( self.encblock(self.cfg.baseChannels, outChannels, kernel=1, padding=0, norm=False) ))
        return toRet


    def addLatent(self, images) :
        if self.noiseInjector is None :
            return images
        latentIn = torch.randn( (images.shape[0], self.noiseInjector[0].in_features), device=images.device )
        latentChannels = self.noiseInjector(latentIn)
        latentChannels, _ = normalizeImages(latentChannels)
        return torch.cat( (images, latentChannels), dim=1 )


    def forward(self, images):
        raise Exception("this is not for direct use")



class Generator(nn.Module):

    def __init__(self, gapW, stripeChannels, bricksChannels, noise=False):
                 #floors, outerKernel=3, noiseChannels=0, links=True, preGenerator=None):
        super().__init__()
        if gapW not in mainFloors.keys() :
            raise Exception(f"Gap width {gapW} is not from the list of possible: {mainFloors.keys()}.")
        floors = mainFloors[gapW]
        if noise :
            self.preGenerator = Generator(gapW, stripeChannels, bricksChannels, noise=False)
        elif gapW == 2 :
            self.preGenerator = None
        else :
            self.preGenerator = Generator(gapW//2, stripeChannels, bricksChannels, noise=False)
        noiseChannels = (1 if noise else 0)
        inChannels = 1 + (0 if self.preGenerator is None else 1)
        self.cfg = DCfgClass(gapW, inChannels, stripeChannels, bricksChannels, False)
        outerKernel=3
        self.bricksGenerator = SubGeneratorTemplate(gapW, True,  inChannels, bricksChannels, stripeChannels,
                                                    floors, outerKernel=outerKernel, noiseChannels=noiseChannels)
        self.stripeGenerator = SubGeneratorTemplate(gapW, False, inChannels, stripeChannels, bricksChannels,
                                                    floors, outerKernel=outerKernel, noiseChannels=noiseChannels)
        deepChans = self.stripeGenerator.postEncoderShape()[1]
        self.deepGenerator = SubGeneratorTemplate(4, False, deepChans, deepChans, 0, deepFloors)
        if not noise :
            self.createLink()


    def createLink(self) :
        bricksSh = self.bricksGenerator.postEncoderShape()
        bricksSz = math.prod(bricksSh[1:])
        self.bricksGenerator.link = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bricksSz, bricksSz//2),
            nn.LeakyReLU(0.2),
            nn.Linear(bricksSz//2, bricksSz),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, bricksSh[1:]),
        )

        encSh = self.stripeGenerator.postEncoderShape()
        chansSz = math.prod(encSh[1:])
        midChans = chansSz * encSh[1] // 2048
        self.stripeGenerator.link = nn.Sequential(
            nn.Conv1d(in_channels=chansSz, out_channels=midChans, kernel_size=1, groups=encSh[1]),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=midChans, out_channels=chansSz, kernel_size=1, groups=encSh[1]),
            nn.LeakyReLU(0.2),
        )

        deepSh = self.deepGenerator.postEncoderShape(inShape=encSh)
        deepSz = math.prod(deepSh[1:])
        self.deepGenerator.link = nn.Sequential(
            nn.Flatten(),
            nn.Linear(deepSz, deepSz),
            nn.LeakyReLU(0.2),
            nn.Linear(deepSz, deepSz),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, deepSh[1:]),
        )


    def fillTheGap(self,images, gap) :
        if images.shape[-2] != gap.shape[-2] or images.shape[0] != gap.shape[0] :
            raise Exception(f"Filling gaps requires inputs of the same size except last dimension. Got {images.shape} and {gap.shape}.")
        if self.cfg.sinoSh[-1] % images.shape[-1] != 0 :
            raise Exception(f"Width of the images {images.shape[-1]} is an integer of {self.cfg.sinoSh[-1]}.")
        ratio = self.cfg.sinoSh[-1] // images.shape[-1]
        gapStart = self.cfg.gapRng[-1].start
        if self.cfg.gapW % ratio + gapStart % ratio != 0 :
            raise Exception(f"Gap width {self.cfg.gapW} and gap start {gapStart} must be integer multiples of {ratio}.")
        gapStart //= ratio
        gapWidth = self.cfg.gapW // ratio
        if images.shape[-1] == gap.shape[-1] :
            gapRng = np.s_[gapStart:gapStart+gapWidth]
        elif gap.shape[-1] == gapWidth :
            gapRng = np.s_[:]
        else :
            raise Exception(f"Bad gap width {gap.shape[-1]} for filling images of width {images.shape[-1]}.")
        channels = min(images.shape[1], gap.shape[1])
        gapped = torch.cat( [ images[:,:channels,:, : gapStart],
                              gap   [:,:channels,:, gapRng].to(images.device),
                              images[:,:channels,:, gapStart+gapWidth : ]
                            ],
                            dim=-1
                          )
        gapped = torch.cat( (gapped, images[:,channels:,...]), dim=1 )
        return gapped


    def preProc(self, images) :
        if self.preGenerator is None :
            return images
        images, orgDims = unsqeeze4dim(images)
        if isinstance(self.preGenerator, GeneratorTemplate) :
            orgSh = images.shape[-2:]
            preSh = self.preGenerator.cfg.sinoSh
            if orgSh != preSh :
                images = images.to(firstDevice(self.preGenerator))
                images = torch.nn.functional.interpolate(images, size=preSh, mode='area')
            res = self.preGenerator.forward(images)
            if orgSh != preSh :
                res = torch.nn.functional.interpolate(res, size=orgSh, mode='bilinear')
        elif self.cfg.gapW == 2:
            images = images.to(firstDevice(self))
            gapStart = self.cfg.gapRng[-1].start
            gapStop  = self.cfg.gapRng[-1].stop
            with torch.no_grad() :
                gap = torch.cat( [ ( 2*images[:,0:1,:,[gapStart-1]] + images[:,0:1,:,[gapStop]   ] ) / 3,
                                   ( 2*images[:,0:1,:,[gapStop]   ] + images[:,0:1,:,[gapStart-1]] ) / 3,
                                 ],
                                 dim=-1
                               )
                res = self.fillTheGap(images, gap)
        else :
            raise Exception("Failed to preproccess. Something is wrong with the generator.")
            #images = images.to(firstDevice(self))
            #with torch.no_grad() :
            #    res = images.clone().detach()
            #    mask = torch.ones_like(res, dtype=torch.bool)
            #    mask[self.cfg.gapRng] = 0
            #    res[self.cfg.gapRng] = 0
            #    res = pytorch_amfill.ops.amfill(res, mask)
        return squeezeOrg(res, orgDims)


    def generateImages(self, images, noises=None) :
        return self.fillTheGap(images, self.forward(images)[:,[0],...])


    def forwardLink(self, images, bricks):

        if self.stripeGenerator.link is None :
            postChans = images
        else :
            tDev = firstDevice(self.stripeGenerator.link)
            postChans = self.stripeGenerator.link( images.to(tDev).view(images.shape[0], -1, 1) ).view(images.shape)

        dwTrain = [images.to(firstDevice(self.deepGenerator)),]
        # encoding
        for level, encoder in enumerate(self.deepGenerator.encoders) :
            dwTrain.append( encoder(dwTrain[-1]) )
        if self.deepGenerator.link is None :
            mid = dwTrain[-1]
        else :
            mid = self.deepGenerator.link(dwTrain[-1])
        upTrain = [mid,]
        # decoding
        for level, decoder in enumerate( self.deepGenerator.decoders) :
            imgsI = torch.cat( [ img.to(firstDevice(self.deepGenerator)) for img in (
                                    upTrain[-1],
                                    dwTrain[-1-level]
                                ) ], dim=1)
            upTrain.append( decoder(imgsI) )
        postDeep = upTrain[-1].to(postChans.device)

        postImages = postChans + postDeep

        if self.bricksGenerator.link is None :
            postBricks = bricks
        else :
            postBricks = self.bricksGenerator.link(bricks)

        return postImages, postBricks


    def forward(self, images):


        # preform inputs
        lrImages = self.preProc(images)
        filledImages = self.fillTheGap(images.to(lrImages.device), lrImages[:,[0],...])
        if self.preGenerator is None :
            stripeIn = filledImages.to(firstDevice(self.stripeGenerator))
        else :
            stripeIn = torch.cat( [ img.to(firstDevice(self.stripeGenerator)) for img in (
                                    filledImages,
                                    lrImages
                                )  ], dim=1)
        stripeIn, stripe_norms = normalizeImages(stripeIn)
        stripeIn = self.stripeGenerator.addLatent(stripeIn)
        stripe_dwTrain = [ self.stripeGenerator.entrance(stripeIn),]
        stripeBricked_dwTrain = [stripe2bricks(stripe_dwTrain[-1]),]

        lrImagesBricked = stripe2bricks(lrImages)
        filledImagesBricked = stripe2bricks(filledImages)
        if self.preGenerator is None :
            bricksIn = lrImagesBricked.to(firstDevice(self.bricksGenerator))
        else :
            bricksIn = torch.cat( [ img.to(firstDevice(self.bricksGenerator)) for img in (
                                    filledImagesBricked,
                                    lrImagesBricked
                                ) ], dim=1)
        bricksIn, bricks_norms = normalizeImages(bricksIn)
        bricksIn = self.bricksGenerator.addLatent(bricksIn)
        bricks_dwTrain = [ self.bricksGenerator.entrance(bricksIn), ]
        bricksStriped_dwTrain = [bricks2stripe(bricks_dwTrain[-1]),]

        # encoding
        for level, (brick_encoder, stripe_encoder) in enumerate( zip(self.bricksGenerator.encoders, self.stripeGenerator.encoders) ):
            bricksI = torch.cat( [bricks_dwTrain[-1],
                                  stripeBricked_dwTrain[-1].to(firstDevice(self.bricksGenerator))
                                 ], dim=1 )
            bricks_dwTrain.append( brick_encoder( bricksI ) )
            stripeI = torch.cat( [stripe_dwTrain[-1],
                                  bricksStriped_dwTrain[-1].to(firstDevice(self.stripeGenerator))
                                 ], dim=1 )
            stripe_dwTrain.append( stripe_encoder(stripeI))
            bricksStriped_dwTrain.append( bricks2stripe(bricks_dwTrain[-1]) )
            stripeBricked_dwTrain.append( stripe2bricks(stripe_dwTrain[-1]) )


        # linking
        stripe_mid, bricks_mid = self.forwardLink(stripe_dwTrain[-1], bricks_dwTrain[-1])
        bricks_upTrain = [bricks_mid,]
        stripe_upTrain = [stripe_mid,]

        # decoding
        for level, (brick_decoder, stripe_decoder) in enumerate( zip(self.bricksGenerator.decoders,
                                                                     self.stripeGenerator.decoders) ):
            bricksI = torch.cat( [ img.to(firstDevice(self.bricksGenerator)) for img in (
                                    bricks_upTrain[-1],
                                    bricks_dwTrain[-1-level],
                                    stripe2bricks(stripe_upTrain[-1]),
                                    stripeBricked_dwTrain[-1-level]
                                ) ], dim=1)
            stripeI = torch.cat( [ img.to(firstDevice(self.stripeGenerator)) for img in (
                                    stripe_upTrain[-1],
                                    stripe_dwTrain[-1-level],
                                    bricks2stripe(bricks_upTrain[-1]),
                                    bricksStriped_dwTrain[-1-level],
                                ) ] , dim=1)
            bricks_upTrain.append( brick_decoder(bricksI) )
            stripe_upTrain.append( stripe_decoder(stripeI) )

        # last touches
        stripeI = torch.cat( [ img.to(firstDevice(self.stripeGenerator)) for img in (
                stripe_upTrain[-1],
                bricks2stripe(bricks_upTrain[-1]),
                stripeIn,
            ) ], dim=1 )
        stripe_results = self.stripeGenerator.lastTouch(stripeI) * self.stripeGenerator.amplitude
        stripe_results = reNormalizeImages(stripe_results, stripe_norms, stdOnly=True)

        bricksI = torch.cat( [ img.to(firstDevice(self.bricksGenerator)) for img in (
                bricks_upTrain[-1],
                stripe2bricks(stripe_upTrain[-1]),
                bricksIn,
            ) ], dim=1 )
        bricks_results = self.bricksGenerator.lastTouch(bricksI) * self.bricksGenerator.amplitude
        bricks_results = reNormalizeImages(bricks_results, bricks_norms, stdOnly=True)
        bricks_results = bricks2stripe(bricks_results)

        # final result
        results = lrImages + bricks_results.to(lrImages.device) + stripe_results.to(lrImages.device)
        return results




class SubDiscriminatorTemplate(SubTemplate):

    def __init__(self, gapW, brick, inChannels, baseChannels, otherChannels, floors, body, outerKernel=3, inShape=None):
        super().__init__(gapW, brick, inChannels, baseChannels, otherChannels, floors, outerKernel=outerKernel, norms=False)
        if body :
            self.body = self.createBody(inShape)

    def createBody(self, inShape) :
        encSh = self.postEncoderShape(inShape=inShape)
        leftChannels = math.prod(encSh)
        layers = [nn.Flatten(),]
        while leftChannels > 1 :
            outChannels = max(leftChannels//4, 1)
            layers.append(nn.Linear(leftChannels, outChannels))
            layers.append( nn.Sigmoid() if outChannels == 1 else  nn.LeakyReLU(0.2) )
            leftChannels = outChannels
        return torch.nn.Sequential(*layers)

    def forward(self, images):
        raise Exception("this is not for direct use")



class Discriminator(nn.Module):

    def __init__(self, gapW, stripeChannels, bricksChannels, fromPair=False):
        super().__init__()
        if gapW not in mainFloors.keys() :
            raise Exception(f"Gap width {gapW} is not from the list of possible: {mainFloors.keys()}.")
        floors = mainFloors[gapW]
        inChannels = (2 if fromPair else 1)
        self.cfg = DCfgClass(gapW, inChannels, stripeChannels, bricksChannels, False)
        self.bricksDiscriminator = SubDiscriminatorTemplate(gapW, True,  inChannels, bricksChannels, stripeChannels,
                                                            floors, body=True, outerKernel=3)
        self.stripeDiscriminator = SubDiscriminatorTemplate(gapW, False, inChannels, stripeChannels, bricksChannels,
                                                            floors, body=False, outerKernel=3)
        postStripeShape = self.stripeDiscriminator.postEncoderShape()
        deepChans = postStripeShape[1]
        self.deepDiscriminator = SubDiscriminatorTemplate(4, False, deepChans, deepChans, 0, deepFloors,
                                                          body=True, inShape=postStripeShape)


    def forward(self, images):

        # preform inputs
        stripeIn, _ = normalizeImages(images.to(firstDevice(self.stripeDiscriminator)))
        stripe_dwTrain = [ self.stripeDiscriminator.entrance(stripeIn),]
        stripeBricked_dwTrain = [stripe2bricks(stripe_dwTrain[-1]),]

        bricksIn = stripe2bricks(images).to(firstDevice(self.bricksDiscriminator))
        bricksIn, _ = normalizeImages(bricksIn)
        bricks_dwTrain = [ self.bricksDiscriminator.entrance(bricksIn), ]
        bricksStriped_dwTrain = [bricks2stripe(bricks_dwTrain[-1]),]

        # encoding
        for level, (brick_encoder, stripe_encoder) in enumerate( zip(self.bricksDiscriminator.encoders, self.stripeDiscriminator.encoders) ):
            bricksI = torch.cat( [bricks_dwTrain[-1],
                                  stripeBricked_dwTrain[-1].to(firstDevice(self.bricksDiscriminator))
                                 ], dim=1 )
            bricks_dwTrain.append( brick_encoder( bricksI ) )
            stripeI = torch.cat( [stripe_dwTrain[-1],
                                  bricksStriped_dwTrain[-1].to(firstDevice(self.stripeDiscriminator))
                                 ], dim=1 )
            stripe_dwTrain.append( stripe_encoder(stripeI))
            bricksStriped_dwTrain.append( bricks2stripe(bricks_dwTrain[-1]) )
            stripeBricked_dwTrain.append( stripe2bricks(stripe_dwTrain[-1]) )

        brickResults = self.bricksDiscriminator.body(bricks_dwTrain[-1]).view(images.shape[0],-1).mean(dim=1).view(-1,1)

        dwTrain = stripe_dwTrain[-1]
        # deep dive
        for encoder in self.deepDiscriminator.encoders :
            dwTrain = encoder(dwTrain)
        stripeResults = self.deepDiscriminator.body(dwTrain).view(images.shape[0],1)

        return ( brickResults + stripeResults ) / 2
        #return stripeResults





def model(gapW, kind, addin, stripeChannels, bricksChannels, modelfile=None) :
    if kind.lower() in [ "g", "gen", "generator" ] :
        modToRet = Generator(gapW, stripeChannels, bricksChannels, noise=addin)
    elif kind.lower() in [ "d", "dis", "discriminator" ] :
        modToRet = Discriminator(gapW, stripeChannels, bricksChannels, fromPair=addin)
    else :
        raise Exception(f"Unknown kind of model {kind}. Can be 'generator' or 'discriminator' ")
    if modelfile is not None :
        modToRet.load_state_dict(torch.load(modelfile, map_location=torch.device('cpu')))
    return modToRet.eval().requires_grad_(False)




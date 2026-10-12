# filename: psColor Module
import math
import cv2
import numpy as np

def Gray2PseudoColor(grayImage, colorMap):
    """

    Args:
        grayImage (_type_): _description_
        colorMap (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Recolor a label image by table lookup: every pixel whose value is label i
    # gets replaced by colorMap[i] (BGR), one label value at a time.
    imageDim = grayImage.shape
    colorImage = np.zeros((imageDim[0],imageDim[1],3), dtype=np.uint8)
    colormapDim = colorMap.shape
    for i in range(colormapDim[0]):
        inds = np.nonzero(grayImage==i)
        colorImage[inds] = (colorMap[i,0],colorMap[i,1],colorMap[i,2])
    return colorImage

def CreateColorMap( colorNum, firstColorBlack = 0):
    """
    A labeled image is hard to look at — labels 1, 2, 3 are nealy identical shades
    of near black. This function builds a lookup table of colorNum distinc RGB colors
    by evaluating 3 sine waves, phase-shifted per channel, so channel 0 ramps up, channel 1 ramps down,
    and channel 2 peaks in the middle as the index i goes from 0 to colorNum - 1. The result is a smooth,
    visually distinc color per index. firstColorBlack = 1 forces index 0 to pure black instead of whatever the sine
    formula gives it,
    """
    colorMap = np.zeros( (colorNum, 3), dtype=np.uint8)
    for i in range(colorNum):
        # Channel 0: ramps up monotonically (sin sweeps 0 -> pi/2 as i runs 0..colorNum-1).
        colorMap[i,0] = int(round(255*abs(math.sin(1.0*i*math.pi/2/colorNum))))
        # Channel 1: ramps down monotonically (same sweep, shifted by pi/2, i.e. cosine).
        colorMap[i,1] = int(round(255*abs(math.sin(1.0*i*math.pi/2/colorNum+math.pi/2))))
        # Channel 2: rises then falls, peaking at the middle index (full half sine period).
        colorMap[i,2] = int(round(255*abs(math.sin(1.0*i*math.pi/colorNum))))
    if firstColorBlack:
        # Force label 0 (typically the background) to pure black.
        colorMap[0,0] = 0
        colorMap[0,1] = 0
        colorMap[0,2] = 0
    return colorMap

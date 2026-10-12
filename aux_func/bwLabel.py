# filename: bwLabel Module
import cv2
import numpy as np

def labeling(bwImage):
    """
    takes a binary image (background = 0, foreground = 255) and turns each separate blob of 
    foreground pixels into its own numbered region — blob 1 gets value 1, blob 2 gets value 2, 
    and so on — instead of every blob sharing the same 255. 
    It does this with cv2.findContours(..., cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE):
    
    - RETR_CCOMP: organizes countours into a 2-level hierarchy — outer boundaries at the
    top level, any holes inside them one level down.
    - CHAIN_APPROX_SIMPLE: compresses straight contour segments to just ther endpoints.

    Args:
        bwImage (array): imagem binária
    """
    if cv2.__version__.startswith('2.'): # so para distinguir as versões do open cv
        bw = bwImage.copy()
        (contourSeq, contourHierarchy) = cv2.findContours(bw, \
                    cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cv2.__version__.startswith('3.'):
        (bw, contourSeq, contourHierarchy) = cv2.findContours(bwImage, \
                    cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cv2.__version__.startswith('4.'):
        (contourSeq, contourHierarchy) = cv2.findContours(bwImage, \
                    cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    grayLevel = 0
    outImage = np.zeros(bwImage.shape)
    idx = 0
    
    # Walk only the top-level contours (outer blobs), skipping holes entirely.
    # contourHierarchy[0][idx] = [next, previous, first_child, parent]; following
    # "next" starting from idx=0 visits every outer blob exactly once, in order.
    while idx >= 0: 
        grayLevel = grayLevel+1
        color = (grayLevel,grayLevel,grayLevel)
        
        # Fill the blob solid (thickness=-1) with its label value. Holes are
        # never drawn separately, so a blob with a hole is filled solid, hole
        # included - this labels connected regions, it doesn't subtract holes.
        cv2.drawContours(outImage, contourSeq, idx, color, -1, 8 , contourHierarchy)
        idx = contourHierarchy[0][idx][0]

    # Return the number of distinct labels (including background 0) and the
    # labeled image itself, where each pixel holds its blob's label value.
    return grayLevel+1, outImage

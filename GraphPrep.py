import numpy as np
import scipy.sparse as sp
from spektral.layers import GCSConv



def getAdjMatrix(size):
    if size == 1:
        return
        
    index = 0
    aMatrix = np.zeros((size*size, size*size))
    pixelDict = {}

    for y in range(size):
            for x in range(size):
                pixelDict[(x,y)] = index
                index = index+1
    for y in range(size):
            for x in range(size):
                #print(NormImages[0][0][x][y])
                if x > 0 and x < size-1 and y > 0 and y < size-1:
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x,y+1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x,y-1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y+1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y+1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y-1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y-1)]] = 1
                elif x == 0:
                    if y == 0:
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x,y+1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y+1)]] = 1
                    elif y == size-1:
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x,y-1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y-1)]] = 1
                    else:
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x,y+1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x,y-1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y-1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y+1)]] = 1
                elif x == size-1:
                    if y == 0:
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x,y+1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y+1)]] = 1
                    elif y == size-1:
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x,y-1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y-1)]] = 1
                    else:
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x,y+1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x,y-1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y-1)]] = 1
                        aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y+1)]] = 1
                elif y == 0:
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x,y+1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y+1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y+1)]] = 1
                elif y == size-1:
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x,y-1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x-1,y-1)]] = 1
                    aMatrix[pixelDict[(x,y)]][pixelDict[(x+1,y-1)]] = 1
    #aMatrix = sp.csr_matrix(aMatrix)
    return aMatrix

def getFeatures(image):
    size = np.array(image).shape[0]
    NormImages=image
    features = {}
    index = 0
    for y in range(size):
            for x in range(size):
                # if isRGB:
                #     features[index] = [float(NormImages[0][y][x]),float(NormImages[1][y][x]),float(NormImages[2][y][x])]
                # else:
                features[index] = [float(NormImages[y][x])]
                index = index+1
    features = np.array(list(features.values()))
    return features

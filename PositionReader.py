import numpy as np
from parameters import Parameters as p

def file_length(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    return i + 1

def readPos():
    posFile = open('Output_Data/POI_Positions.txt', 'r')

    count = 1
    coordMat = []
    lineCount = file_length('Output_Data/POI_Positions.txt')

    for line in posFile:
        for coord in line.split('\t'):
            if (coord != '\n') and (count == lineCount):
                coordMat.append(float(coord))
        count += 1

    return np.reshape(coordMat, (p.num_pois, 2))

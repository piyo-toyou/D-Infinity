# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:13:46 2020

@author: mgkog
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot, cm, colors

#input parameters
csv_path = "Nishiharamura_FD_5m_33.csv"
test_path = "C:/Users/S2212357/Documents/Z9_Training/QGIS/WaterFlow/FlowDirection/test_csv.csv"
df = pd.read_csv(csv_path, sep=",", header=None, index_col=None)
myarrayFD = df.values
returnarrayFA = np.full(myarrayFD.shape, -0.5)

def Zero(arr, i, j):
    if 135 < arr[i,j+1] < 225 or 180 < arr[i-1,j+1] < 270 or 225 < arr[i-1,j] < 315 \
            or 270 < arr[i-1,j-1] or 315 < arr[i,j-1] or arr[i,j-1] < 45 or 0 < arr[i+1,j-1] < 90 \
            or 45 < arr[i+1,j] < 135 or 90 < arr[i+1,j+1] < 180: 
        return -1.0
    else:
        return 0

def ACM(arr1, arr2, i, j):
    d = 0
    # east
    if arr1[i,j+1] <= 135 or arr1[i,j+1] >= 225:
        pass
    elif 135 < arr1[i,j+1] < 180 and arr2[i,j+1] > 0:
        d += (arr2[i,j+1]+1)*((arr1[i,j+1]-135)/45)
    elif 135 < arr1[i,j+1] < 180 and -0.6 < arr2[i,j+1] <= 0:
        d += (arr1[i,j+1]-135)/45
    elif 180 <= arr1[i,j+1] < 225 and arr2[i,j+1] > 0:
        d += (arr2[i,j+1]+1)*((225-arr1[i,j+1])/45)
    elif 180 <= arr1[i,j+1] < 225 and -0.6 < arr2[i,j+1] <= 0:
        d += (225-arr1[i,j+1])/45
    else:
        return False
    
    #north - east
    if arr1[i-1,j+1] <= 180 or arr1[i-1,j+1] >= 270:
        pass
    elif 180 < arr1[i-1,j+1] < 225 and arr2[i-1,j+1] > 0:
        d += (arr2[i-1,j+1]+1)*((arr1[i-1,j+1]-180)/45)
    elif 180 < arr1[i-1,j+1] < 225 and -0.6 < arr2[i-1,j+1] <= 0:
        d += (arr1[i-1,j+1]-180)/45
    elif 225 <= arr1[i-1,j+1] < 270 and arr2[i-1,j+1] > 0:
        d += (arr2[i-1,j+1]+1)*((270-arr1[i-1,j+1])/45)
    elif 225 <= arr1[i-1,j+1] < 270 and -0.6 < arr2[i-1,j+1] <= 0:
        d += (270-arr1[i-1,j+1])/45
    else:
        return False
    
    # north
    if arr1[i-1,j] <= 225 or arr1[i-1,j] >= 315:
        pass
    elif 225 < arr1[i-1,j] < 270 and arr2[i-1,j] > 0:
        d += (arr2[i-1,j]+1)*((arr1[i-1,j]-225)/45)
    elif 225 < arr1[i-1,j] < 270 and -0.6 < arr2[i-1,j] <= 0:
        d += (arr1[i-1,j]-225)/45
    elif 270 <= arr1[i-1,j] < 315 and arr2[i-1,j] > 0:
        d += (arr2[i-1,j]+1)*((315-arr1[i-1,j])/45)
    elif 270 <= arr1[i-1,j] < 315 and -0.6 < arr2[i-1,j] <= 0:
        d += (315-arr1[i-1,j])/45
    else:
        return False

    # north - west
    if arr1[i-1,j-1] <= 270:
        pass
    elif 270 < arr1[i-1,j-1] < 315 and arr2[i-1,j-1] > 0:
        d += (arr2[i-1,j-1]+1)*((arr1[i-1,j-1]-270)/45)
    elif 270 < arr1[i-1,j-1] < 315 and -0.6 < arr2[i-1,j-1] <= 0:
        d += (arr1[i-1,j-1]-270)/45
    elif 315 <= arr1[i-1,j-1] and arr2[i-1,j-1] > 0:
        d += (arr2[i-1,j-1]+1)*((360-arr1[i-1,j-1])/45)
    elif 315 <= arr1[i-1,j-1] and -0.6 < arr2[i-1,j-1] <= 0:
        d += (360-arr1[i-1,j-1])/45
    else:
        return False

    # west
    if arr1[i,j-1] >= 45 and arr1[i,j-1] <= 315:
        pass
    elif 0 <= arr1[i,j-1] < 45 and arr2[i,j-1] > 0:
        d += (arr2[i,j-1]+1)*(1-(arr1[i,j-1]/45))
    elif 0 <= arr1[i,j-1] < 45 and -0.6 < arr2[i,j-1] <= 0:
        d += 1-(arr1[i,j-1]/45)
    elif 315 < arr1[i,j-1] and arr2[i,j-1] > 0:
        d += (arr2[i,j-1]+1)*((arr1[i,j-1]-315)/45)
    elif 315 < arr1[i,j-1] and -0.6 < arr2[i,j-1] <= 0:
        d += (arr1[i,j-1]-315)/45
    else:
        return False

    # south - west
    if arr1[i+1,j-1] >= 90 or arr1[i+1,j-1] <= 0:
        pass
    elif arr1[i+1,j-1] < 45 and arr2[i+1,j-1] > 0:
        d += (arr2[i+1,j-1]+1)*(arr1[i+1,j-1]/45)
    elif arr1[i+1,j-1] < 45 and -0.6 < arr2[i+1,j-1] <= 0:
        d += arr1[i+1,j-1]/45
    elif 45 <= arr1[i+1,j-1] < 90 and arr2[i+1,j-1] > 0:
        d += (arr2[i+1,j-1]+1)*((90-arr1[i+1,j-1])/45)
    elif 45 <= arr1[i+1,j-1] < 90 and -0.6 < arr2[i+1,j-1] <= 0:
        d += (90-arr1[i+1,j-1])/45
    else:
        return False
    
    # south
    if arr1[i+1,j] <= 45 or arr1[i+1,j] >= 135:
        pass
    elif 45 < arr1[i+1,j] < 90 and arr2[i+1,j] > 0:
        d += (arr2[i+1,j]+1)*((arr1[i+1,j]-45)/45)
    elif 45 < arr1[i+1,j] < 90 and -0.6 < arr2[i+1,j] <= 0:
        d += (arr1[i+1,j]-45)/45
    elif 90 <= arr1[i+1,j] < 135 and arr2[i+1,j] > 0:
        d += (arr2[i+1,j]+1)*((135-arr1[i+1,j])/45)
    elif 90 <= arr1[i+1,j] < 135 and -0.6 < arr2[i+1,j] <= 0:
        d += (135-arr1[i+1,j])/45
    else:
        return False

    # south - east
    if arr1[i+1,j+1] <= 90 or arr1[i+1,j+1] >= 180:
        pass
    elif 90 < arr1[i+1,j+1] < 135 and arr2[i+1,j+1] > 0:
        d += (arr2[i+1,j+1]+1)*((arr1[i+1,j+1]-90)/45)
    elif 90 < arr1[i+1,j+1] < 135 and -0.6 < arr2[i+1,j+1] <= 0:
        d += (arr1[i+1,j+1]-90)/45
    elif 135 <= arr1[i+1,j+1] < 180 and arr2[i+1,j+1] > 0:
        d += (arr2[i+1,j+1]+1)*((180-arr1[i+1,j+1])/45)
    elif 135 <= arr1[i+1,j+1] < 180 and -0.6 < arr2[i+1,j+1] <= 0:
        d += (180-arr1[i+1,j+1])/45
    else:
        return False

    return d
        
for i, j in np.argwhere(myarrayFD != -1):
    returnarrayFA[i][j] = Zero(myarrayFD, i, j)

loop = 0


while True:
    for i, j in np.argwhere(returnarrayFA == -1):
        f = ACM(myarrayFD, returnarrayFA, i, j)
        if f:
            returnarrayFA[i][j] = f
        else:
            pass
    loop += 1
    if loop % 100 == 0:
        print(loop)
    if loop == 1000:
        break

pyplot.imshow(returnarrayFA, cmap=cm.cool)
pyplot.colorbar(shrink=.92)
pyplot.show()
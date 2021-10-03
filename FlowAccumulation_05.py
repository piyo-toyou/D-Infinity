# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:13:46 2020

@author: mgkog
"""

import numpy as np

myarray = np.array(([135.00, 264.20, 225.00],
                    [270.00, 225.00, 205.96],
                    [225.00, 191.80, 180.00]))

#define dx dy
rows = myarray.shape[0]
cols = myarray.shape[1]
returnarrayFA = np.zeros((rows, cols))

def Zero(arr, i, j):
    if i-1 >= 0 and j-1 >= 0 and i+1 < rows and j+1 < cols:
        if arr[i-1][j-1] < -999 or arr[i-1][j+1] < -999 or arr[i+1][j-1] < -999 or arr[i+1][j+1] < -999:
            return -0.5
        else:
            if 135 < arr[i,j+1] < 225 or 180 < arr[i-1,j+1] < 270 or 225 < arr[i-1,j] < 315 \
                    or 270 < arr[i-1,j-1] < 360 or 315 < arr[i,j-1] <= 360 or 0 <= arr[i,j-1] < 45 or 0 < arr[i+1,j-1] < 90 \
                    or 45 < arr[i+1,j] < 135 or 90 < arr[i+1,j+1] < 180: 
    
                return -1.0
            
            else:
                return 0
    else:
        return -0.5

def ACM(arr1, arr2, i, j):
    if i-1 >= 0 and j-1 >= 0 and i+1 < rows and j+1 < cols:
        if arr2[i][j] != -1:
            return False
        else:
            d = 0
            if arr1[i,j+1] <= 135 or arr1[i,j+1] >= 225:
                pass
            elif 135 < arr1[i,j+1] < 180 and arr2[i,j+1] > 0:
                d += (arr2[i,j+1]+1)*(1-((180-arr1[i,j+1])/45))
            elif 135 < arr1[i,j+1] < 180 and -0.6 < arr2[i,j+1] <= 0:
                d += 1-((180-arr1[i,j+1])/45)
            elif 180 == arr1[i,j+1] and arr2[i,j+1] > 0:
                d += arr2[i,j+1]+1
            elif 180 == arr1[i,j+1] and -0.6 < arr2[i,j+1] <= 0:
                d += 1
            elif 180 < arr1[i,j+1] < 225 and arr2[i,j+1] > 0:
                d += (arr2[i,j+1]+1)*((225-arr1[i,j+1])/45)
            elif 180 < arr1[i,j+1] < 225 and -0.6 < arr2[i,j+1] <= 0:
                d += (225-arr1[i,j+1])/45
            else:
                return False

            if arr1[i-1,j+1] <= 180 or arr1[i-1,j+1] >= 270:
                pass
            elif 180 < arr1[i-1,j+1] < 225 and arr2[i-1,j+1] > 0:
                d += (arr2[i-1,j+1]+1)*(1-((225-arr1[i-1,j+1])/45))
            elif 180 < arr1[i-1,j+1] < 225 and -0.6 < arr2[i-1,j+1] <= 0:
                d += 1-((225-arr1[i-1,j+1])/45)
            elif 225 == arr1[i-1,j+1] and arr2[i-1,j+1] > 0:
                d += arr2[i-1,j+1]+1
            elif 225 == arr1[i-1,j+1] and -0.6 < arr2[i-1,j+1] <= 0:
                d += 1
            elif 225 < arr1[i-1,j+1] < 270 and arr2[i-1,j+1] > 0:
                d += (arr2[i-1,j+1]+1)*((270-arr1[i-1,j+1])/45)
            elif 225 < arr1[i-1,j+1] < 270 and -0.6 < arr2[i-1,j+1] <= 0:
                d += (270-arr1[i-1,j+1])/45
            else:
                return False
            
            if arr1[i-1,j] <= 225 or arr1[i-1,j] >= 315:
                pass
            elif 225 < arr1[i-1,j] < 270 and arr2[i-1,j] > 0:
                d += (arr2[i-1,j]+1)*(1-((270-arr1[i-1,j])/45))
            elif 225 < arr1[i-1,j] < 270 and -0.6 < arr2[i-1,j] <= 0:
                d += 1-((270-arr1[i-1,j])/45)
            elif 270 == arr1[i-1,j] and arr2[i-1,j] > 0:
                d += arr2[i-1,j]+1
            elif 270 == arr1[i-1,j] and -0.6 < arr2[i-1,j] <= 0:
                d += 1
            elif 270 < arr1[i-1,j] < 315 and arr2[i-1,j] > 0:
                d += (arr2[i-1,j]+1)*((315-arr1[i-1,j])/45)
            elif 270 < arr1[i-1,j] < 315 and -0.6 < arr2[i-1,j] <= 0:
                d += (315-arr1[i-1,j])/45
            else:
                return False

            if arr1[i-1,j-1] <= 270:
                pass
            elif 270 < arr1[i-1,j-1] < 315 and arr2[i-1,j-1] > 0:
                d += (arr2[i-1,j-1]+1)*(1-((315-arr1[i-1,j-1])/45))
            elif 270 < arr1[i-1,j-1] < 315 and -0.6 < arr2[i-1,j-1] <= 0:
                d += 1-((315-arr1[i-1,j-1])/45)
            elif 315 == arr1[i-1,j-1] and arr2[i-1,j-1] > 0:
                d += arr2[i-1,j-1]+1
            elif 315 == arr1[i-1,j-1] and -0.6 < arr2[i-1,j-1] <= 0:
                d += 1
            elif 315 < arr1[i-1,j-1] <= 360 and arr2[i-1,j-1] > 0:
                d += (arr2[i-1,j-1]+1)*((360-arr1[i-1,j-1])/45)
            elif 315 < arr1[i-1,j-1] <= 360 and -0.6 < arr2[i-1,j-1] <= 0:
                d += (360-arr1[i-1,j-1])/45
            else:
                return False

            if 45 <= arr1[i,j-1] <= 315:
                pass
            elif 0 < arr1[i,j-1] < 45 and arr2[i,j-1] > 0:
                d += (arr2[i,j-1]+1)*((45-arr1[i,j-1])/45)
            elif 0 < arr1[i,j-1] < 45 and -0.6 < arr2[i,j-1] <= 0:
                d += (45-arr1[i,j-1])/45
            elif 0 == arr1[i,j-1] and arr2[i,j-1] > 0:
                d += arr2[i,j-1]+1
            elif 0 == arr1[i,j-1] and -0.6 < arr2[i,j-1] <= 0:
                d += 1
            elif 315 < arr1[i,j-1] and arr2[i,j-1] > 0:
                d += (arr2[i,j-1]+1)*(1-((360-arr1[i,j-1])/45))
            elif 315 < arr1[i,j-1] and -0.6 < arr2[i,j-1] <= 0:
                d += 1-(360-arr1[i,j-1])/45
            else:
                return False

            if arr1[i+1,j-1] >= 90 or 0 >= arr1[i+1,j-1]:
                pass
            elif 0 < arr1[i+1,j-1] < 45 and arr2[i+1,j-1] > 0:
                d += (arr2[i+1,j-1]+1)*(1-((45-arr1[i+1,j-1])/45))
            elif 0 < arr1[i+1,j-1] < 45 and -0.6 < arr2[i+1,j-1] <= 0:
                d += 1-((45-arr1[i+1,j-1])/45)
            elif 45 == arr1[i+1,j-1] and arr2[i+1,j-1] > 0:
                d += arr2[i+1,j-1]+1
            elif 45 == arr1[i+1,j-1] and -0.6 < arr2[i+1,j-1] <= 0:
                d += 1
            elif 45 < arr1[i+1,j-1] < 90 and arr2[i+1,j-1] > 0:
                d += (arr2[i+1,j-1]+1)*((90-arr1[i+1,j-1])/45)
            elif 45 < arr1[i+1,j-1] < 90 and -0.6 < arr2[i+1,j-1] <= 0:
                d += (90-arr1[i+1,j-1])/45
            else:
                return False

            if arr1[i+1,j] <= 45 or arr1[i+1,j] >= 135:
                pass
            elif 45 < arr1[i+1,j] < 90 and arr2[i+1,j] > 0:
                d += (arr2[i+1,j]+1)*(1-((90-arr1[i+1,j])/45))
            elif 45 < arr1[i+1,j] < 90 and -0.6 < arr2[i+1,j] <= 0:
                d += 1-((90-arr1[i+1,j])/45)
            elif 90 == arr1[i+1,j] and arr2[i+1,j] > 0:
                d += arr2[i+1,j]+1
            elif 90 == arr1[i+1,j] and -0.6 < arr2[i+1,j] <= 0:
                d += 1
            elif 90 < arr1[i+1,j] < 135 and arr2[i+1,j] > 0:
                d += (arr2[i+1,j]+1)*((135-arr1[i+1,j])/45)
            elif 90 < arr1[i+1,j] < 135 and -0.6 < arr2[i+1,j] <= 0:
                d += (135-arr1[i+1,j])/45
            else:
                return False

            if arr1[i+1,j+1] <= 90 or arr1[i+1,j+1] >= 180:
                pass
            elif 90 < arr1[i+1,j+1] < 135 and arr2[i+1,j+1] > 0:
                d += (arr2[i+1,j+1]+1)*(1-((135-arr1[i+1,j+1])/45))
            elif 90 < arr1[i+1,j+1] < 135 and -0.6 < arr2[i+1,j+1] <= 0:
                d += 1-((135-arr1[i+1,j+1])/45)
            elif 135 == arr1[i+1,j+1] and arr2[i+1,j+1] >= 0:
                d += arr2[i+1,j+1]+1
            elif 135 == arr1[i+1,j+1] and -0.6 < arr2[i+1,j+1] <= 0:
                d += 1
            elif 135 < arr1[i+1,j+1] < 180 and arr2[i+1,j+1] > 0:
                d += (arr2[i+1,j+1]+1)*((180-arr1[i+1,j+1])/45)
            elif 135 < arr1[i+1,j+1] < 180 and -0.6 < arr2[i+1,j+1] <= 0:
                d += (180-arr1[i+1,j+1])/45
            else:
                return False

            return d

    
    else:
        return False
        
"""
for i in range(0, rows):
    for j in range(0, cols):
        returnarrayFA[i][j] = Zero(myarray, i, j)
"""

returnarrayFA = np.array(([0, 0, 28.5],
                          [0.10, -1, 32.3],
                          [-1, -1, 10.4]))

loop = 0

while True:
    print(loop)
    for i in range(0, rows):
        for j in range(0, cols):
            f = ACM(myarray, returnarrayFA, i, j)
            if f:
                returnarrayFA[i][j] = f
            else:
                pass
    loop += 1

    print(loop)
    for i in range(rows-1, -1, -1):
        for j in range(cols-1, -1, -1):
            f = ACM(myarray, returnarrayFA, i, j)
            if f:
                returnarrayFA[i][j] = f
            else:
                pass
    loop += 1
    if -1 not in returnarrayFA or loop > 10:
        break

print(returnarrayFA)



import numpy as np
from numpy.lib.function_base import copy
import pandas as pd
from matplotlib import pyplot, cm, colors
from math import degrees, atan, pi, sqrt

csv_path = "C:/Users/S2212357/Documents/Z9_Training/QGIS/WaterFlow/DEM/FG-GML-4930-17-DEM5A/Nishiharamura_ClipDEM_5m.csv"
df = pd.read_csv(csv_path, sep=",", header=0,index_col=0)
myarray = df.values

dx = 5
dy = 5
dxy = sqrt(pow(dx,2)+pow(dy,2))
dtan = atan(dx/dy)
Ysize = myarray.shape[0]
Xsize = myarray.shape[1]
returnarrayS = np.zeros((Ysize, Xsize))
returnarrayD = np.ones((Ysize, Xsize))
returnarrayF = np.zeros((Ysize, Xsize))

def DInfinity(arr, i, j):
    try:
        b = [[0,1],[1,-1],[1,1],[2,-1],[2,1],[3,-1],[3,1],[4,-1]]
        
        s1en = (arr[i,j]-arr[i,j+1])/dx
        s2en = (arr[i,j+1]-arr[i-1,j+1])/dy
        s0en = (arr[i,j]-arr[i-1,j+1])/dxy
        s1ne = (arr[i,j]-arr[i-1,j])/dx 
        s2ne = (arr[i-1,j]-arr[i-1,j+1])/dy
        s1nw = (arr[i,j]-arr[i-1,j])/dx
        s2nw = (arr[i-1,j]-arr[i-1,j-1])/dy
        s0nw = (arr[i,j]-arr[i-1,j-1])/dxy
        s1wn = (arr[i,j]-arr[i,j-1])/dx
        s2wn = (arr[i,j-1]-arr[i-1,j-1])/dy
        s1ws = (arr[i,j]-arr[i,j-1])/dx
        s2ws = (arr[i,j-1]-arr[i+1,j-1])/dy
        s0ws = (arr[i,j]-arr[i+1,j-1])/dxy
        s1sw = (arr[i,j]-arr[i+1,j])/dx
        s2sw = (arr[i+1,j]-arr[i+1,j-1])/dy
        s1se = (arr[i,j]-arr[i+1,j])/dx
        s2se = (arr[i+1,j]-arr[i+1,j+1])/dy
        s0se = (arr[i,j]-arr[i+1,j+1])/dxy
        s1es = (arr[i,j]-arr[i,j+1])/dx
        s2es = (arr[i,j+1]-arr[i+1,j+1])/dy

        #1 east -> north
        if s1en > 0:
            ren_temp = atan(s2en/s1en)
            if ren_temp > dtan:
                ren = dtan; sen = (arr[i,j]-arr[i-1,j+1])/dxy
            elif ren_temp > 0:
                ren = ren_temp; sen = sqrt(pow(s1en,2)+pow(s2en,2))
            else:
                ren = 0.00; sen = s1en
        else:
            if  s0en > 0:
                ren = dtan; sen = s0en
            else:
                ren = 0.00; sen = s1en

        #2 north -> east
        if s1ne > 0:
            rne_temp = atan(s2ne/s1ne)
            if rne_temp > dtan:
                rne = dtan; sne = (arr[i,j]-arr[i-1,j+1])/dxy
            elif rne_temp > 0:
                rne = rne_temp; sne = sqrt(pow(s1ne,2)+pow(s2ne,2))
            else:
                rne = 0.00; sne = s1ne
        else:
            if s0en > 0:
                rne = dtan; sne = s0en
            else:
                rne = 0.00; sne = s1ne

        #3 north -> west
        if s1nw > 0:
            rnw_temp = atan(s2nw/s1nw)
            if rnw_temp > dtan:
                rnw = dtan; snw = (arr[i,j]-arr[i-1,j-1])/dxy
            elif rnw_temp > 0:
                rnw = rnw_temp; snw = sqrt(pow(s1nw,2)+pow(s2nw,2))
            else:
                rnw = 0.00; snw = s1nw
        else:
            if  s0nw > 0:
                rnw = dtan; snw = s0nw
            else:
                rnw = 0.00; snw = s1nw

        #4 west -> north
        if s1wn > 0:
            rwn_temp = atan(s2wn/s1wn)
            if rwn_temp > dtan:
                rwn = dtan; swn = (arr[i,j]-arr[i-1,j-1])/dxy
            elif rwn_temp > 0:
                rwn = rwn_temp; swn = sqrt(pow(s1wn,2)+pow(s2wn,2))
            else:
                rwn = 0.00; swn = s1wn
        else:
            if s0nw > 0:
                rwn = dtan; swn = s0nw
            else:
                rwn = 0.00; swn = s1wn

        #5 west -> south
        if s1ws > 0:
            rws_temp = atan(s2ws/s1ws)
            if rws_temp > dtan:
                rws = dtan; sws = (arr[i,j]-arr[i+1,j-1])/dxy
            elif rws_temp > 0:
                rws = rws_temp; sws = sqrt(pow(s1ws,2)+pow(s2ws,2))
            else:
                rws = 0.00; sws = s1ws
        else:
            if s0ws > 0:
                rws = dtan; sws = s0ws
            else:
                rws = 0.00; sws = s1ws

        #6 south -> west
        if s1sw > 0:
            rsw_temp = atan(s2sw/s1sw)
            if rsw_temp > dtan:
                rsw = dtan; ssw = (arr[i,j]-arr[i+1,j-1])/dxy
            elif rsw_temp > 0:
                rsw = rsw_temp; ssw = sqrt(pow(s1sw,2)+pow(s2sw,2))
            else:
                rsw = 0.00; ssw = s1sw
        else:
            if  s0ws > 0:
                rsw = dtan; ssw = s0ws
            else:
                rsw = 0.00; ssw = s1sw

        #7 south -> east
        if s1se > 0:
            rse_temp = atan(s2se/s1se)
            if rse_temp > dtan:
                rse = dtan; sse = (arr[i,j]-arr[i+1,j+1])/dxy
            elif rse_temp > 0:
                rse = rse_temp; sse = sqrt(pow(s1se,2)+pow(s2se,2))
            else:
                rse = 0.00; sse = s1se
        else:
            if s0se > 0:
                rse = dtan; sse = s0se
            else:
                rse = 0.00; sse = s1se

        #8 east -> south
        if s1es > 0:
            res_temp = atan(s2es/s1es)
            if res_temp > dtan:
                res = dtan; ses = (arr[i,j]-arr[i+1,j+1])/dxy
            elif res_temp > 0:
                res = res_temp; ses = sqrt(pow(s1es,2)+pow(s2es,2))
            else:
                res = 0.00; ses = s1es
        else:
            if s0se > 0:
                res = dtan; ses = s0se
            else:
                res = 0.00; ses = s1es
        
        r_tuple = (ren, rne, rnw, rwn, rws, rsw, rse, res)
        s_tuple = (sen, sne, snw, swn, sws, ssw, sse, ses)
        
        smax = max(s_tuple)
        if smax > 0:
            sdeg = degrees(atan(smax))
            sid = s_tuple.index(smax)
            r = r_tuple[sid]
            rg = degrees(b[sid][1]*r+b[sid][0]*pi/2)
        else:
            sdeg = 0.00
            rg = -1
        return sdeg, rg
    except:
        return np.nan, np.nan

def Sink(dem, flag, dinf, i, j):
    dem_copy = copy(dem)
    flag_check = np.array((0, 1))
    if any((w in Around(flag, i, j) for w in flag_check)):
        pass

def Flat(arr, i, j):
    pass

def Flag(arr, i, j):
    if i-1 >= 0 and j-1 >= 0 and i+1 < Ysize and j+1 < Xsize:
        if (Around(arr, i, j) < np.array((-999))).any():
            return np.nan
        try:
            h0 = arr[i][j]
            h_around = Around(arr, i, j)
            if (h0 > h_around).any():
                return 0 #Dinfinity
            elif (h0 == h_around).any():
                return 1 #Flat
            else:
                return 2 #Sink
        except:
            return np.nan #Error
    else:
        return np.nan

def Around(arr, i, j):
    h_above = np.array((arr[i-1][j-1:j+2]))
    h_mid = np.array((arr[i][j-1], arr[i][j+1]))
    h_below = np.array((arr[i+1][j-1:j+2]))
    return np.hstack((h_above, h_mid, h_below))

#calculation
for i, j in np.ndindex(myarray.shape):
    returnarrayF[i][j] = Flag(myarray, i, j)

for i, j in np.ndindex(myarray.shape):
    if returnarrayF[i][j] == 0:
        args = DInfinity(myarray, i, j)
        returnarrayS[i][j] = args[0]
        returnarrayD[i][j] = args[1]
    else:
        pass

# np.set_printoptions(precision=1)
# print(returnarrayD)

cmap = cm.cool
cmap_data = cmap(np.arange(cmap.N))
cmap_data[0, 3] = 0
custom_cool = colors.ListedColormap(cmap_data)
pyplot.imshow(returnarrayD, cmap=custom_cool)
pyplot.colorbar(shrink=.92)
pyplot.show()
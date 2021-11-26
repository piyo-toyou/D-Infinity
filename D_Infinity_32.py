
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
returnarrayS = np.zeros(myarray.shape)
returnarrayD = np.ones(myarray.shape)
returnarrayF = np.zeros(myarray.shape)

def Around(arr, i, j):
    h_above = np.array((arr[i-1][j-1:j+2]))
    h_mid = np.array((arr[i][j-1], np.nan, arr[i][j+1]))
    h_below = np.array((arr[i+1][j-1:j+2]))
    return np.vstack((h_above, h_mid, h_below))

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

def DInfinity(ij):
    i, j = ij
    try:
        b = [[0,1],[1,-1],[1,1],[2,-1],[2,1],[3,-1],[3,1],[4,-1]]
        
        s1en = (myarray[i,j]-myarray[i,j+1])/dx
        s2en = (myarray[i,j+1]-myarray[i-1,j+1])/dy
        s0en = (myarray[i,j]-myarray[i-1,j+1])/dxy
        s1ne = (myarray[i,j]-myarray[i-1,j])/dx 
        s2ne = (myarray[i-1,j]-myarray[i-1,j+1])/dy
        s1nw = (myarray[i,j]-myarray[i-1,j])/dx
        s2nw = (myarray[i-1,j]-myarray[i-1,j-1])/dy
        s0nw = (myarray[i,j]-myarray[i-1,j-1])/dxy
        s1wn = (myarray[i,j]-myarray[i,j-1])/dx
        s2wn = (myarray[i,j-1]-myarray[i-1,j-1])/dy
        s1ws = (myarray[i,j]-myarray[i,j-1])/dx
        s2ws = (myarray[i,j-1]-myarray[i+1,j-1])/dy
        s0ws = (myarray[i,j]-myarray[i+1,j-1])/dxy
        s1sw = (myarray[i,j]-myarray[i+1,j])/dx
        s2sw = (myarray[i+1,j]-myarray[i+1,j-1])/dy
        s1se = (myarray[i,j]-myarray[i+1,j])/dx
        s2se = (myarray[i+1,j]-myarray[i+1,j+1])/dy
        s0se = (myarray[i,j]-myarray[i+1,j+1])/dxy
        s1es = (myarray[i,j]-myarray[i,j+1])/dx
        s2es = (myarray[i,j+1]-myarray[i+1,j+1])/dy

        #1 east -> north
        if s1en > 0:
            ren_temp = atan(s2en/s1en)
            if ren_temp > dtan:
                ren = dtan; sen = (myarray[i,j]-myarray[i-1,j+1])/dxy
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
                rne = dtan; sne = (myarray[i,j]-myarray[i-1,j+1])/dxy
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
                rnw = dtan; snw = (myarray[i,j]-myarray[i-1,j-1])/dxy
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
                rwn = dtan; swn = (myarray[i,j]-myarray[i-1,j-1])/dxy
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
                rws = dtan; sws = (myarray[i,j]-myarray[i+1,j-1])/dxy
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
                rsw = dtan; ssw = (myarray[i,j]-myarray[i+1,j-1])/dxy
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
                rse = dtan; sse = (myarray[i,j]-myarray[i+1,j+1])/dxy
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
                res = dtan; ses = (myarray[i,j]-myarray[i+1,j+1])/dxy
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
        return i, j, sdeg, rg
    except:
        return i, j, np.nan, np.nan

def Dinfinity_Receive(args):
    i, j, s, d = args
    returnarrayS[i][j] = s
    returnarrayD[i][j] = d

def Sink(dem, flag, dinf, i, j):
    dem_copy = copy(dem)
    target_area = np.zeros((Ysize, Xsize), dtype=int)
    target_area[i][j] = 1
    my_around = Around(flag, i, j)
    if any((1 in my_around)):
        ind_temp = np.argwhere(my_around == 1)
        pass
    else:
        dinf[i][j] = 300

def Flat(arr, i, j):
    pass


#calculation
#Flag
for i, j in np.ndindex(myarray.shape):
    returnarrayF[i][j] = Flag(myarray, i, j)

#Regular cells
args = map(DInfinity, np.argwhere(returnarrayF == 0))
map(Dinfinity_Receive, args)

#Sink cells
#for i, j in np.argwhere(returnarrayF == 2):
#    Sink(myarray, returnarrayF, returnarrayD, i, j)

#Visualization
cmap = cm.cool
cmap_data = cmap(np.arange(cmap.N))
cmap_data[0, 3] = 0
custom_cool = colors.ListedColormap(cmap_data)
pyplot.imshow(returnarrayD, cmap=custom_cool)
pyplot.colorbar(shrink=.92)
pyplot.show()
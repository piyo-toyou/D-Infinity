
import numpy as np
import pandas as pd
from matplotlib import pyplot, cm, colors
from math import degrees, atan, pi, sqrt

csv_path = "Nishiharamura_Clip2DEM_5m_Fill.csv"
df = pd.read_csv(csv_path, sep=",", header=0,index_col=0)
myarray = df.values

dx = 5
dy = 5
dxy = sqrt(pow(dx,2)+pow(dy,2))
dtan = atan(dx/dy)
Ysize = myarray.shape[0]
Xsize = myarray.shape[1]
returnarrayS = np.zeros(myarray.shape)
returnarrayD = np.full(myarray.shape, -1.0)
returnarrayF = np.zeros(myarray.shape)

def Around(arr, X):
    if X.size == 2:
        i, j = X
        h_above = arr[i-1][j-1:j+2]
        h_mid = np.array((arr[i][j-1], 9999, arr[i][j+1]))
        h_below = arr[i+1][j-1:j+2]
        return np.vstack((h_above, h_mid, h_below))
    else:
        temp_around = np.arange(3)
        for x1, x2 in enumerate(X):
            X_d1 = X - X[x1] # 対象範囲との被りを検出
            X_d2 = X_d1[np.all(-1<=X_d1, axis=1)]
            X_d3 = X_d2[np.all(X_d2<=1, axis=1)]

            i, j = x2 # 周囲の標高値を抜き出し
            h_above = arr[i-1][j-1:j+2]
            h_mid = np.array((arr[i][j-1], 9999, arr[i][j+1]))
            h_below = arr[i+1][j-1:j+2]
            h_marge = np.vstack((h_above, h_mid, h_below))
            for s, t in X_d3:
                h_marge[s+1, t+1] = 9999
            temp_around = np.vstack((temp_around, h_marge))
        return temp_around[1:]

def Flag(arr, i, j):
    if i-1 >= 0 and j-1 >= 0 and i+1 < Ysize and j+1 < Xsize:
        if (Around(arr, np.array((i, j))) < np.array((-999))).any():
            return np.nan
        try:
            h0 = arr[i][j]
            h_around = Around(arr, np.array((i, j)))
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
    arr = myarray
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
        return i, j, sdeg, rg
    except:
        return i, j, np.nan, np.nan

def Dinfinity_Receive(args):
    i, j, s, d = args
    returnarrayS[i][j], returnarrayD[i][j] = s, d

def D8(t_area, t_idx, t_point, out_point): # 対象範囲全体、番号、座標と、流出点座標
    out_check1 = out_point - t_point
    if -1 <= out_check1[0] <= 1 and -1 <= out_check1[1] <= 1:
        t = np.squeeze(out_check1)
    else:
        T_d1 = np.delete(t_area, t_idx, 0) # target diffrence
        T_d1 = T_d1 - t_point
        T_d1 = T_d1[np.all(-1<=T_d1, axis=1)]
        T_d1 = T_d1[np.all(T_d1<=1, axis=1)] # 隣接する範囲を探索
        if T_d1.size == 2: # 1箇所と接する場合
            t = np.squeeze(T_d1)
        else: # 複数個所と接する場合
            GT_d1 = T_d1 + t_point # global target diffrence
            GT_d2 = GT_d1 - out_point
            GT_d3 = np.array([np.linalg.norm(i) for i in GT_d2])
            t = T_d1[np.argmin(GT_d3)]
    if t[0] == -1:
        d8 = 90 + t[1] * -45
    elif t[0] == 0:
        d8 = 90 + t[1] * -90
    else:
        d8 = 270 + t[1] * 45
    return d8

def Sink(dem, flag, dinf, i, j):
    dem_copy = dem.copy()
    target_area = np.array((i, j))
    my_around = Around(dem, target_area)
    for p in range(10):
        # 複数点を捉えるために、np.whereを使用する方が厳密
        # 対象領域周囲から、最小標高点を探す
        min_value = np.min(my_around)
        min_idx = np.unravel_index(np.argmin(my_around), my_around.shape)
        if not p:   GMI = np.array((i, j)) + min_idx - np.array((1, 1)) # global min index
        else:       GMI = target_area[min_idx[0]//3] + ((min_idx[0]%3, min_idx[1])) - np.array((1, 1))
        target_area = np.vstack((target_area, GMI))
        flag[GMI[0], GMI[1]] = 1
        for u, v in target_area:
            dem_copy[u, v] = min_value # 窪地埋め処理
        my_around = Around(dem_copy, target_area) # 対象領域の周囲を更新
        if (my_around < min_value).any(): # 周囲の点から流出点を探す
            print(i, j, p, "break")
            break
    out_idx = np.unravel_index(np.argmin(my_around), my_around.shape)
    GOI = target_area[out_idx[0]//3] + ((out_idx[0]%3, out_idx[1])) - np.array((1, 1)) # global out index
    flag[GOI[0], GOI[1]] = 1.5
    for q1, q2 in enumerate(target_area):
        dinf[q2[0], q2[1]] = D8(target_area, q1, q2, GOI)

def Flat(arr, i, j):
    pass

#calculation
#Flag
for i, j in np.ndindex(myarray.shape):
    returnarrayF[i][j] = Flag(myarray, i, j)

#Regular cells
args = map(DInfinity, np.argwhere(returnarrayF == 0))
for x in args:  Dinfinity_Receive(x)

#Sink cells
for i, j in np.argwhere(returnarrayF == 2):
    Sink(myarray, returnarrayF, returnarrayD, i, j)

#Visualization
cmap = cm.cool
cmap_data = cmap(np.arange(cmap.N))
cmap_data[0, 3] = 0
custom_cool = colors.ListedColormap(cmap_data)
pyplot.imshow(returnarrayD, cmap=custom_cool)
pyplot.colorbar(shrink=.92)
pyplot.show()

"""
out_df  = pd.DataFrame(returnarrayD)
out_df.to_csv("Nishiharamura_FD_5m_33.csv", header=None, index=None)
"""
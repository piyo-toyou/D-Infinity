
import numpy as np
import pandas as pd
from matplotlib import pyplot, cm, colors
from math import degrees, atan, pi, sqrt
import copy

csv_path = "Nishiharamura_Clip5DEM_5m_Fill.csv"
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
    out_check = out_point - t_point
    if -1 <= out_check[0] <= 1 and -1 <= out_check[1] <= 1:
        t = np.squeeze(out_check)
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
            GT_d3 = np.array([np.linalg.norm(i) for i in GT_d2]) # 流出点への距離を算出
            t = T_d1[np.argmin(GT_d3)] # 最も距離が短くなる隣接点に流す
    if t[0] == -1:
        d8 = 90 + t[1] * -45
    elif t[0] == 0:
        d8 = 90 + t[1] * -90
    else:
        d8 = 270 + t[1] * 45
    return d8

#探索関数：ゴールしたらそのときの位置・移動数を返す
def Maze(pos, ml, rt):
    #スタート位置（x座標, y座標, 移動回数）をセット
    while len(pos) > 0:#探索可能ならTrue
        y, x, depth, origin = pos.pop(0) #リストから探索する位置を取得

        #ゴールについた時点で終了
        if ml[y][x] == 1:
            rt[y][x] = origin
            return [(y, x), depth, origin]

        #探索済みとしてセット
        ml[y][x] = 2

        #現在位置の上下左右を探索：〇<2は壁でもなく探索済みでもないものを示す
        if ml[y-1][x] < 2:#上
            pos.append([y-1, x, depth + 1, 1000 * y + x])
            if rt[y-1][x] == 0:
                rt[y-1][x] = 1000 * y + x
        if ml[y+1][x] < 2:#下
            pos.append([y+1, x, depth + 1, 1000 * y + x])
            if rt[y+1][x] == 0:
                rt[y+1][x] = 1000 * y + x
        if ml[y][x+1] < 2:#右
            pos.append([y, x+1, depth + 1, 1000 * y + x])
            if rt[y][x+1] == 0:
                rt[y][x+1] = 1000 * y + x
        if ml[y][x-1] < 2:#左
            pos.append([y, x-1, depth + 1, 1000 * y + x])
            if rt[y][x-1] == 0:
                rt[y][x-1] = 1000 * y + x
        if ml[y+1][x-1] < 2:#左下
            pos.append([y+1, x-1, depth + 1, 1000 * y + x])
            if rt[y+1][x-1] == 0:
                rt[y+1][x-1] = 1000 * y + x
        if ml[y-1][x-1] < 2:#左上
            pos.append([y-1, x-1, depth + 1, 1000 * y + x])
            if rt[y-1][x-1] == 0:
                rt[y-1][x-1] = 1000 * y + x
        if ml[y-1][x+1] < 2:#右上
            pos.append([y-1, x+1, depth + 1, 1000 * y + x])
            if rt[y-1][x+1] == 0:
                rt[y-1][x+1] = 1000 * y + x
        if ml[y+1][x+1] < 2:#右下
            pos.append([y+1, x+1, depth + 1, 1000 * y + x])
            if rt[y+1][x+1] == 0:
                rt[y+1][x+1] = 1000 * y + x
    return False

def SimpleD8(p):
    if p[0] == -1:
        SD8 = 90 + p[1] * -45
    elif p[0] == 0:
        SD8 = 90 + p[1] * -90
    else:
        SD8 = 270 + p[1] * 45
    return SD8

def ReverseOrder(goal, st, dir, rt):
    ori_y, ori_x = goal[0][0], goal[0][1] # 元となる位置 origin
    rev_y, rev_x = goal[2]//1000, goal[2]%1000 # 逆順を辿った位置 reverse
    while True:
        if rev_y == st[0][0] and rev_x == st[0][1]:
            relativ_position = [ori_y - st[0][0],ori_x - st[0][1]]
            sd8 = SimpleD8(relativ_position)
            dir[rev_y][rev_x] = sd8
            break
        relativ_position = [ori_y - rev_y, ori_x - rev_x]
        sd8 = SimpleD8(relativ_position)
        dir[rev_y][rev_x] = sd8
        ori_y, ori_x = rev_y, rev_x
        rev_y, rev_x = rt[rev_y][rev_x]//1000, rt[rev_y][rev_x]%1000

def BFS(ta, goi, arrD):
    mazeYedge = np.min((goi[0], np.min(ta.T[0])))
    mazeXedge = np.min((goi[1], np.min(ta.T[1])))
    mazeY1 = np.max((goi[0], np.max(ta.T[0])))- mazeYedge + 1
    mazeX1 = np.max((goi[1], np.max(ta.T[1])))- mazeXedge + 1
    maze = np.full((mazeY1, mazeX1), 9, dtype=np.int8)
    direction = np.full((mazeY1+2, mazeX1+2), 9, dtype=np.int16)
    for xy in ta-np.array((mazeYedge, mazeXedge)): # 対象範囲を白抜き
        maze[xy[0], xy[1]] = 0
        direction[xy[0]+1, xy[1]+1] = 10
    maze[goi[0]-mazeYedge, goi[1]-mazeXedge] = 1 # 流出点を設定
    direction[goi[0]-mazeYedge+1, goi[1]-mazeXedge+1] = 1
    maze = np.vstack((np.full((mazeX1,), 9), maze, np.full((mazeX1,), 9))) # 周囲を埋め
    maze = np.hstack((np.full((mazeY1+2,1), 9), maze, np.full((mazeY1+2,1), 9)))
    route = maze.astype(np.int32) # 道順を格納する配列を用意

    while True:
        if np.argwhere(direction == 10).size:
            maze_list = maze.tolist()
            route = maze.astype(np.int32) # 道順を格納する配列を初期化
            ij = np.argwhere(direction == 10)
            i, j = ij[np.random.choice(ij.shape[0],1)][0]
            start = [[i, j, 0, 1000*i+j]]     #スタート位置
            start_copy = copy.copy(start)

            result = Maze(start_copy, maze_list, route)  #探索
            print(result)
            ReverseOrder(result, start, direction, route)
            print("end")
        else:
            print(direction)
            for ij in np.argwhere(maze == 0):
                i, j = ij
                arrD[i+mazeYedge-1][j+mazeXedge-1] = direction[i][j]
            break

def Flat(dem, flag, dinf, i, j):
    target_area = np.array((i, j))
    while True:
        flag_around = Around(flag, target_area)
        if 1 in flag_around:
            flat_area = np.where(flag_around == 1)
            flat_area_idx = np.vstack((flat_area[0], flat_area[1])).T
            for s in range(len(flat_area_idx)):
                if target_area.size == 2:
                    G = target_area
                else:
                    G = target_area[flat_area_idx[s][0]//3]
                F = ((flat_area_idx[s][0]%3, flat_area_idx[s][1]))
                I = np.array((1, 1))
                GFI = G + F - I # global flat index
                target_area = np.vstack((target_area, GFI))
        else:
            break
    if target_area.size >= 4:
        target_area = np.unique(target_area, axis=0)
    my_around = Around(dem, target_area)
    for p in range(100):
        # 複数点を捉えるために、np.whereを使用する方が厳密
        # 対象領域周囲から、最小標高点を探す
        min_value = np.min(my_around)
        min_idx = np.unravel_index(np.argmin(my_around), my_around.shape)
        if target_area.size == 2:   GMI = target_area + min_idx - np.array((1, 1)) # global min index
        else:       GMI = target_area[min_idx[0]//3] + ((min_idx[0]%3, min_idx[1])) - np.array((1, 1))
        if 0 in GMI or Ysize-1 == GMI[0] or Xsize-1 == GMI[1]:
            print(i, j, p, "edge break")
            break
        target_area = np.vstack((target_area, GMI))
        for u, v in target_area:
            dem[u, v] = min_value # 窪地埋め処理
        my_around = Around(dem, target_area) # 対象領域の周囲を更新
        if (my_around < min_value).any(): # 周囲の点から流出点を探す
            print(i, j, p, "break")
            break
    out_idx = np.unravel_index(np.argmin(my_around), my_around.shape)
    GOI = target_area[out_idx[0]//3] + ((out_idx[0]%3, out_idx[1])) - np.array((1, 1)) # global out index
    try:
        for q1, q2 in enumerate(target_area):
            flag[q2[0], q2[1]] = 0.75
        BFS(target_area, GOI, dinf)
    except:
        GOI = target_area + out_idx - np.array((1, 1))
        target_area = target_area[np.newaxis, :]
        for q1, q2 in enumerate(target_area):
            dinf[q2[0], q2[1]] = D8(target_area, q1, q2, GOI)
            flag[q2[0], q2[1]] = 0.75

def Sink(dem, flag, dinf, i, j):
    target_area = np.array((i, j))
    my_around = Around(dem, target_area)
    for p in range(100):
        # 複数点を捉えるために、np.whereを使用する方が厳密
        # 対象領域周囲から、最小標高点を探す
        min_value = np.min(my_around)
        min_idx = np.unravel_index(np.argmin(my_around), my_around.shape)
        if not p:   GMI = np.array((i, j)) + min_idx - np.array((1, 1)) # global min index
        else:       GMI = target_area[min_idx[0]//3] + ((min_idx[0]%3, min_idx[1])) - np.array((1, 1))
        if 0 in GMI or Ysize-1 == GMI[0] or Xsize-1 == GMI[1]:
            print(i, j, p, "edge break")
            break
        target_area = np.vstack((target_area, GMI))
        for u, v in target_area:
            dem[u, v] = min_value # 窪地埋め処理
        my_around = Around(dem, target_area) # 対象領域の周囲を更新
        if (my_around < min_value).any(): # 周囲の点から流出点を探す
            print(i, j, p, "break")
            break
    out_idx = np.unravel_index(np.argmin(my_around), my_around.shape)
    GOI = target_area[out_idx[0]//3] + ((out_idx[0]%3, out_idx[1])) - np.array((1, 1)) # global out index
    try:
        for q1, q2 in enumerate(target_area):
            flag[q2[0], q2[1]] = 1.75
        BFS(target_area, GOI, dinf)
    except:
        GOI = target_area + out_idx - np.array((1, 1))
        target_area = target_area[np.newaxis, :]
        for q1, q2 in enumerate(target_area):
            dinf[q2[0], q2[1]] = D8(target_area, q1, q2, GOI)
            flag[q2[0], q2[1]] = 1.75
    print(i, j, GOI)



#calculation
#Flag
for i, j in np.ndindex(myarray.shape):
    returnarrayF[i][j] = Flag(myarray, i, j)

#Regular cells
args = map(DInfinity, np.argwhere(returnarrayF == 0))
for x in args:  Dinfinity_Receive(x)

#Flat cells
while True:
    if np.argwhere(returnarrayF == 1).size:
        ij = np.argwhere(returnarrayF == 1)
        i, j = ij[np.random.choice(ij.shape[0],1)][0]
    else:
        break
    Flat(myarray, returnarrayF, returnarrayD, i, j)

#Sink cells
while True:
    if np.argwhere(returnarrayF == 2).size:
        ij = np.argwhere(returnarrayF == 2)
        i, j = ij[np.random.choice(ij.shape[0],1)][0]
    else:
        break
    Sink(myarray, returnarrayF, returnarrayD, i, j)

#Visualization
cmap = cm.cool
cmap_data = cmap(np.arange(cmap.N))
cmap_data[0, 3] = 0
custom_cool = colors.ListedColormap(cmap_data)
pyplot.imshow(returnarrayF, cmap=custom_cool)
pyplot.colorbar(shrink=.92)
pyplot.show()

"""
out_df  = pd.DataFrame(returnarrayD)
out_df.to_csv("Nishiharamura_FD_5m_37.csv", header=None, index=None)
# """

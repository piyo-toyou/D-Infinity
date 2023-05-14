import numpy as np
import pandas as pd
from matplotlib import pyplot, cm, colors
from math import degrees, atan, pi, sqrt
import copy

csv_path = "Nishiharamura_DEM_5m_Fill.csv"
df = pd.read_csv(csv_path, sep=",", header=0, index_col=0)
myarray = df.values

cell_size_x = 5
cell_size_y = 5
cell_size_xy = sqrt(pow(cell_size_x, 2) + pow(cell_size_y, 2))
dtan = atan(cell_size_x / cell_size_y)
Ysize, Xsize = myarray.shape
returnarrayS = np.zeros(myarray.shape)
returnarrayD = np.full(myarray.shape, -1.0)
returnarrayF = np.zeros(myarray.shape)
returnarrayF[[0, -1], :] = np.nan
returnarrayF[:, [0, -1]] = np.nan

def Around(input_array, X):
    if X.size == 2:
        i, j = X
        h_above = input_array[i-1][j-1:j+2]
        h_mid = np.array((input_array[i][j-1], 9999, input_array[i][j+1]))
        h_below = input_array[i+1][j-1:j+2]
        return np.vstack((h_above, h_mid, h_below))
    else:
        uniq_idx = np.unique(X, axis=0, return_index=True)[1]
        X_unique = [X[uniq_idx] for uniq_idx in sorted(uniq_idx)]
        temp_around = np.arange(3)
        for x1, x2 in enumerate(X_unique):
            X_d1 = X_unique - X_unique[x1] # 対象範囲との被りを検出
            X_d2 = X_d1[np.all(-1<=X_d1, axis=1)]
            X_d3 = X_d2[np.all(X_d2<=1, axis=1)]

            i, j = x2 # 周囲の標高値を抜き出し
            h_above = input_array[i-1][j-1:j+2]
            h_mid = np.array((input_array[i][j-1], 9999, input_array[i][j+1]))
            h_below = input_array[i+1][j-1:j+2]
            h_marge = np.vstack((h_above, h_mid, h_below))
            for s, t in X_d3:
                h_marge[s+1, t+1] = 9999
            temp_around = np.vstack((temp_around, h_marge))
        return temp_around[1:]

def Flag(input_array, i, j):
    if i-1 >= 0 and j-1 >= 0 and i+1 < Ysize and j+1 < Xsize:
        if (Around(input_array, np.array((i, j))) < np.array((-999))).any():
            return np.nan
        try:
            h0 = input_array[i][j]
            h_around = Around(input_array, np.array((i, j)))
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

def calculate_slope_and_flow(s1, s2, s0, dtan):
    if s1 > 0:
        r_temp = atan(s2 / s1)
        if r_temp > dtan:
            slope_degree = dtan
            flow_direction = s0
        elif r_temp > 0:
            slope_degree = r_temp
            flow_direction = sqrt(pow(s1, 2) + pow(s2, 2))
        else:
            slope_degree = 0.00
            flow_direction = s1
    else:
        if s0 > 0:
            slope_degree = dtan
            flow_direction = s0
        else:
            slope_degree = 0.00
            flow_direction = s1
    return slope_degree, flow_direction

def calculate_dInfinity(ij, input_array):
    i, j = ij
    try:
        b = [[0, 1], [1, -1], [1, 1], [2, -1], [2, 1], [3, -1], [3, 1], [4, -1]]

        s1en = (input_array[i, j] - input_array[i, j + 1]) / cell_size_x
        s2en = (input_array[i, j + 1] - input_array[i - 1, j + 1]) / cell_size_y
        s0en = (input_array[i, j] - input_array[i - 1, j + 1]) / cell_size_xy
        s1ne = (input_array[i, j] - input_array[i - 1, j]) / cell_size_x
        s2ne = (input_array[i - 1, j] - input_array[i - 1, j + 1]) / cell_size_y
        s1nw = (input_array[i, j] - input_array[i - 1, j]) / cell_size_x
        s2nw = (input_array[i - 1, j] - input_array[i - 1, j - 1]) / cell_size_y
        s0nw = (input_array[i, j] - input_array[i - 1, j - 1]) / cell_size_xy
        s1wn = (input_array[i, j] - input_array[i, j - 1]) / cell_size_x
        s2wn = (input_array[i, j - 1] - input_array[i - 1, j - 1]) / cell_size_y
        s1ws = (input_array[i, j] - input_array[i, j - 1]) / cell_size_x
        s2ws = (input_array[i, j - 1] - input_array[i + 1, j - 1]) / cell_size_y
        s0ws = (input_array[i, j] - input_array[i + 1, j - 1]) / cell_size_xy
        s1sw = (input_array[i, j] - input_array[i + 1, j]) / cell_size_x
        s2sw = (input_array[i + 1, j] - input_array[i + 1, j - 1]) / cell_size_y
        s1se = (input_array[i, j] - input_array[i + 1, j]) / cell_size_x
        s2se = (input_array[i + 1, j] - input_array[i + 1, j + 1]) / cell_size_y
        s0se = (input_array[i, j] - input_array[i + 1, j + 1]) / cell_size_xy
        s1es = (input_array[i, j] - input_array[i, j + 1]) / cell_size_x
        s2es = (input_array[i, j + 1] - input_array[i + 1, j + 1]) / cell_size_y

        flow_en, slope_en = calculate_slope_and_flow(s1en, s2en, s0en, dtan) #1 east -> north
        flow_ne, slope_ne = calculate_slope_and_flow(s1ne, s2ne, s0en, dtan) #2 north -> east
        flow_nw, slope_nw = calculate_slope_and_flow(s1nw, s2nw, s0nw, dtan) #3 north -> west
        flow_wn, slope_wn = calculate_slope_and_flow(s1wn, s2wn, s0nw, dtan) #4 west -> north
        flow_ws, slope_ws = calculate_slope_and_flow(s1ws, s2ws, s0ws, dtan) #5 west -> south
        flow_sw, slope_sw = calculate_slope_and_flow(s1sw, s2sw, s0ws, dtan) #6 south -> west
        flow_se, slope_se = calculate_slope_and_flow(s1se, s2se, s0se, dtan) #7 south -> east
        flow_es, slope_es = calculate_slope_and_flow(s1es, s2es, s0se, dtan) #8 east -> south

        slope_values = [slope_en, slope_ne, slope_nw, slope_wn, slope_ws, slope_sw, slope_se, slope_es]
        flow_values = [flow_en, flow_ne, flow_nw, flow_wn, flow_ws, flow_sw, flow_se, flow_es]

        smax = max(slope_values)
        if smax > 0:
            slope_degree = degrees(atan(smax))
            sid = slope_values.index(smax)
            r = flow_values[sid]
            flow_direction = degrees(b[sid][1] * r + b[sid][0] * pi / 2)
        else:
            slope_degree = 0.00
            flow_direction = -1
        return {"i": i, "j": j, "sd": slope_degree, "fd": flow_direction}
    except IndexError as ie:
        print(f"List index error occurred at index ({i}, {j}): {ie}")
        return {"i":i, "j":j, "sd":np.nan, "fd":np.nan}
    except Exception as e:
        print(f"Error occurred at index ({i}, {j}): {e}")
        return {"i":i, "j":j, "sd":np.nan, "fd":np.nan}

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

def SimpleD8(p):
    if p[0] == -1:
        SD8 = 90 + p[1] * -45
    elif p[0] == 0:
        SD8 = 90 + p[1] * -90
    else:
        SD8 = 270 + p[1] * 45
    return SD8

#探索関数：ゴールしたらそのときの位置・移動数を返す
#2020/12/30 @Yuya Shimizu
def Maze(pos, ml, rt):
    #スタート位置（y座標, x座標, 移動回数, 方向記憶）をセット
    while len(pos) > 0:#探索可能ならTrue
        y, x, depth, origin = pos.pop(0) #リストから探索する位置を取得

        #ゴールについた時点で終了
        if ml[y][x] == 1:
            rt[y][x] = origin
            return [(y, x), depth, origin]

        #探索済みとしてセット
        if ml[y][x] == 0:
            ml[y][x] = 2
        elif ml[y][x] == 2:
            continue


        #現在位置の上下左右と斜めを探索：〇<2は壁でもなく探索済みでもないものを示す
        if ml[y-1][x] < 2:#上
            pos.append([y-1, x, depth + 1, 10000 * y + x])
            if rt[y-1][x] == 0:
                rt[y-1][x] = 10000 * y + x
        if ml[y+1][x] < 2:#下
            pos.append([y+1, x, depth + 1, 10000 * y + x])
            if rt[y+1][x] == 0:
                rt[y+1][x] = 10000 * y + x
        if ml[y][x+1] < 2:#右
            pos.append([y, x+1, depth + 1, 10000 * y + x])
            if rt[y][x+1] == 0:
                rt[y][x+1] = 10000 * y + x
        if ml[y][x-1] < 2:#左
            pos.append([y, x-1, depth + 1, 10000 * y + x])
            if rt[y][x-1] == 0:
                rt[y][x-1] = 10000 * y + x
        if ml[y+1][x-1] < 2:#左下
            pos.append([y+1, x-1, depth + 1, 10000 * y + x])
            if rt[y+1][x-1] == 0:
                rt[y+1][x-1] = 10000 * y + x
        if ml[y-1][x-1] < 2:#左上
            pos.append([y-1, x-1, depth + 1, 10000 * y + x])
            if rt[y-1][x-1] == 0:
                rt[y-1][x-1] = 10000 * y + x
        if ml[y-1][x+1] < 2:#右上
            pos.append([y-1, x+1, depth + 1, 10000 * y + x])
            if rt[y-1][x+1] == 0:
                rt[y-1][x+1] = 10000 * y + x
        if ml[y+1][x+1] < 2:#右下
            pos.append([y+1, x+1, depth + 1, 10000 * y + x])
            if rt[y+1][x+1] == 0:
                rt[y+1][x+1] = 10000 * y + x
    return False

def ReverseOrder(goal, st, dir, rt):
    ori_y, ori_x = goal[0][0], goal[0][1] # 元となる位置 origin
    rev_y, rev_x = goal[2]//10000, goal[2]%10000 # 逆順を辿った位置 reverse
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
        rev_y, rev_x = rt[rev_y][rev_x]//10000, rt[rev_y][rev_x]%10000

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
            start = [[i, j, 0, 10000*i+j]]     #スタート位置
            start_copy = copy.copy(start)

            result = Maze(start_copy, maze_list, route)  #探索
            ReverseOrder(result, start, direction, route)
        else:
            print("BFS", ta, goi)
            print(direction)
            for ij in np.argwhere(maze == 0):
                i, j = ij
                arrD[i+mazeYedge-1][j+mazeXedge-1] = direction[i][j]
            break

def calculate_flat(dem, flag, dinf, i, j):
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
            target_area = np.unique(target_area, axis=0)
        else:
            break
    if target_area.size >= 4:
        target_area = np.unique(target_area, axis=0)
    my_around = Around(dem, target_area)
    for p in range(1000):
        # 複数点を捉えるために、np.whereを使用する方が厳密
        # 対象領域周囲から、最小標高点を探す
        min_value = np.min(my_around)
        min_idx = np.unravel_index(np.argmin(my_around), my_around.shape)
        if target_area.size == 2:   GMI = target_area + min_idx - np.array((1, 1)) # global min index
        else:       GMI = target_area[min_idx[0]//3] + ((min_idx[0]%3, min_idx[1])) - np.array((1, 1))
        if 0 in GMI or Ysize-1 == GMI[0] or Xsize-1 == GMI[1]:
            print(i, j, p, "edge break Flat")
            break
        target_area = np.vstack((target_area, GMI))
        for u, v in target_area:
            dem[u, v] = min_value # 窪地埋め処理
        my_around = Around(dem, target_area) # 対象領域の周囲を更新
        if (my_around < min_value).any(): # 周囲の点から流出点を探す
            print(i, j, p, "break Flat")
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

def calculate_sink(dem, flag, dinf, i, j):
    target_area = np.array((i, j))
    my_around = Around(dem, target_area)
    for p in range(1000):
        # 複数点を捉えるために、np.whereを使用する方が厳密
        # 対象領域周囲から、最小標高点を探す
        
        min_value = np.min(my_around)
        min_idx = np.unravel_index(np.argmin(my_around), my_around.shape)
        if not p:   GMI = np.array((i, j)) + min_idx - np.array((1, 1)) # global min index
        else:       GMI = target_area[min_idx[0]//3] + ((min_idx[0]%3, min_idx[1])) - np.array((1, 1))
        if 0 in GMI or Ysize-1 == GMI[0] or Xsize-1 == GMI[1]:
            print(i, j, p, "edge break Sink")
            break
        target_area = np.vstack((target_area, GMI))
        target_area = np.unique(target_area, axis=0)
        for u, v in target_area:
            dem[u, v] = min_value # 窪地埋め処理
        my_around = Around(dem, target_area) # 対象領域の周囲を更新
        if (my_around < min_value).any(): # 周囲の点から流出点を探す
            print(i, j, p, "break Sink")
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

# Calculation
#Flag
for i, j in np.ndindex(myarray.shape):
    returnarrayF[i][j] = Flag(myarray, i, j)
print("calculate flag; done.")

#Regular cells
args = map(lambda ij: calculate_dInfinity(ij, input_array=myarray), np.argwhere(returnarrayF == 0))
for result in args:
    i, j = result["i"], result["j"]
    returnarrayS[i][j], returnarrayD[i][j] = result["sd"], result["fd"]
print("calculate dInfinity; done.")

#Flat cells
while True:
    if np.argwhere(returnarrayF == 1).size:
        ij = np.argwhere(returnarrayF == 1)
        i, j = ij[np.random.choice(ij.shape[0],1)][0] #任意の平地セルを対象として平地処理を繰り返す
    else:
        break
    calculate_flat(myarray, returnarrayF, returnarrayD, i, j)
print("calculate flat; done.")

#Sink cells
while True:
    if np.argwhere(returnarrayF == 2).size:
        ij = np.argwhere(returnarrayF == 2)
        i, j = ij[np.random.choice(ij.shape[0],1)][0] #任意の窪地セルを対象として窪地処理を繰り返す
    else:
        break
    calculate_sink(myarray, returnarrayF, returnarrayD, i, j)
print("calculate sink; done.")

# Visualization
cmap = cm.cool
cmap_data = cmap(np.arange(cmap.N))
cmap_data[0, 3] = 0
custom_cool = colors.ListedColormap(cmap_data)
pyplot.imshow(returnarrayD, cmap=custom_cool)
pyplot.colorbar(shrink=.92)
pyplot.show()

# out_df = pd.DataFrame(returnarrayD)
# out_df.to_csv("Nishiharamura_FD_5m_GPT01.csv", header=None, index=None)
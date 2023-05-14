
import numpy as np # 必要ライブラリをインポート
import pandas as pd
from matplotlib import pyplot, cm, colors
from math import degrees, atan, pi, sqrt

csv_path = "Nishiharamura_DEM_5m_Fill.csv" # 入力csvのパスを指定
df = pd.read_csv(csv_path, sep=",", header=0,index_col=0) # csvをデータフレームで読み込み
myarray = df.values # 配列に変換

cell_size_x = 5
cell_size_y = 5
cell_size_xy = sqrt(pow(cell_size_x,2)+pow(cell_size_y,2))
dtan = atan(cell_size_x/cell_size_y)
Ysize = myarray.shape[0]
Xsize = myarray.shape[1]
returnarrayS = np.zeros(myarray.shape)
returnarrayD = np.full(myarray.shape, -1.0)
returnarrayF = np.zeros(myarray.shape)
returnarrayF[0, :] = np.nan
returnarrayF[-1, :] = np.nan
returnarrayF[:, 0] = np.nan
returnarrayF[:, -1] = np.nan

def calculate_slope_and_flow(s1, s2, s0, dtan):
    if s1 > 0:
        r_temp = atan(s2 / s1)
        if r_temp > dtan:
            r = dtan
            s = s0
        elif r_temp > 0:
            r = r_temp
            s = sqrt(pow(s1, 2) + pow(s2, 2))
        else:
            r = 0.00
            s = s1
    else:
        if s0 > 0:
            r = dtan
            s = s0
        else:
            r = 0.00
            s = s1
    return r, s

def calculate_dInfinity(ij, input_array):
    i, j = ij
    try:
        b = [[0,1],[1,-1],[1,1],[2,-1],[2,1],[3,-1],[3,1],[4,-1]]
        
        s1en = (input_array[i,j]-input_array[i,j+1])/cell_size_x
        s2en = (input_array[i,j+1]-input_array[i-1,j+1])/cell_size_y
        s0en = (input_array[i,j]-input_array[i-1,j+1])/cell_size_xy
        s1ne = (input_array[i,j]-input_array[i-1,j])/cell_size_x 
        s2ne = (input_array[i-1,j]-input_array[i-1,j+1])/cell_size_y
        s1nw = (input_array[i,j]-input_array[i-1,j])/cell_size_x
        s2nw = (input_array[i-1,j]-input_array[i-1,j-1])/cell_size_y
        s0nw = (input_array[i,j]-input_array[i-1,j-1])/cell_size_xy
        s1wn = (input_array[i,j]-input_array[i,j-1])/cell_size_x
        s2wn = (input_array[i,j-1]-input_array[i-1,j-1])/cell_size_y
        s1ws = (input_array[i,j]-input_array[i,j-1])/cell_size_x
        s2ws = (input_array[i,j-1]-input_array[i+1,j-1])/cell_size_y
        s0ws = (input_array[i,j]-input_array[i+1,j-1])/cell_size_xy
        s1sw = (input_array[i,j]-input_array[i+1,j])/cell_size_x
        s2sw = (input_array[i+1,j]-input_array[i+1,j-1])/cell_size_y
        s1se = (input_array[i,j]-input_array[i+1,j])/cell_size_x
        s2se = (input_array[i+1,j]-input_array[i+1,j+1])/cell_size_y
        s0se = (input_array[i,j]-input_array[i+1,j+1])/cell_size_xy
        s1es = (input_array[i,j]-input_array[i,j+1])/cell_size_x
        s2es = (input_array[i,j+1]-input_array[i+1,j+1])/cell_size_y

        ren, sen = calculate_slope_and_flow(s1en, s2en, s0en, dtan)
        rne, sne = calculate_slope_and_flow(s1ne, s2ne, s0en, dtan)
        rnw, snw = calculate_slope_and_flow(s1nw, s2nw, s0nw, dtan)
        rwn, swn = calculate_slope_and_flow(s1wn, s2wn, s0nw, dtan)
        rws, sws = calculate_slope_and_flow(s1ws, s2ws, s0ws, dtan)
        rsw, ssw = calculate_slope_and_flow(s1sw, s2sw, s0ws, dtan)
        rse, sse = calculate_slope_and_flow(s1se, s2se, s0se, dtan)
        res, ses = calculate_slope_and_flow(s1es, s2es, s0se, dtan)
        
        r_tuple = (ren, rne, rnw, rwn, rws, rsw, rse, res)
        s_tuple = (sen, sne, snw, swn, sws, ssw, sse, ses)
        
        smax = max(s_tuple)
        if smax > 0:
            slope_degree = degrees(atan(smax))
            sid = s_tuple.index(smax)
            r = r_tuple[sid]
            flow_direction = degrees(b[sid][1]*r+b[sid][0]*pi/2)
        else:
            slope_degree = 0.00
            flow_direction = -1
        return {"i":i, "j":j, "sd":slope_degree, "fd":flow_direction}
    except Exception as e:
        print(f"Error occurred at index ({i}, {j}): {e}")
        return {"i":i, "j":j, "sd":np.nan, "fd":np.nan}

#calculation
args = map(lambda ij: calculate_dInfinity(ij, input_array=myarray), np.argwhere(returnarrayF == 0))
for result in args:
    i, j = result["i"], result["j"]
    returnarrayS[i][j], returnarrayD[i][j] = result["sd"], result["fd"]

#Visualization
cmap = cm.cool
cmap_data = cmap(np.arange(cmap.N))
cmap_data[0, 3] = 0
custom_cool = colors.ListedColormap(cmap_data)
pyplot.imshow(returnarrayD, cmap=custom_cool)
pyplot.colorbar(shrink=.92)
pyplot.show()


out_df  = pd.DataFrame(returnarrayD)
out_df.to_csv("Nishiharamura_FD_5m_57.csv", header=None, index=None)


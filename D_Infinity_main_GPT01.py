import numpy as np
import pandas as pd
from matplotlib import pyplot, cm, colors
from math import degrees, atan, pi, sqrt

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

# Calculation
args = map(lambda ij: calculate_dInfinity(ij, input_array=myarray), np.argwhere(returnarrayF == 0))
for result in args:
    i, j = result["i"], result["j"]
    returnarrayS[i][j], returnarrayD[i][j] = result["sd"], result["fd"]

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
"""
2020/12/30
@Yuya Shimizu
@Wataru Ogurahata edit

番兵
迷路探索１（幅優先探索）
"""

import numpy as np
import copy


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

GOI = np.array((13, 22))
returnarrayD = np.full((38, 39), -1.0)
target_area = np.array([[12, 17],
       [13, 17],
       [14, 17],
       [14, 18],
       [15, 17],
       [15, 18],
       [15, 21],
       [15, 22],
       [16, 17],
       [16, 18],
       [16, 21],
       [16, 22],
       [17, 16],
       [17, 17],
       [17, 18],
       [17, 19],
       [17, 20],
       [17, 21],
       [18, 16],
       [18, 17],
       [18, 18],
       [18, 19],
       [18, 20],
       [18, 21],
       [19, 16],
       [19, 17],
       [19, 18],
       [19, 19],
       [19, 20],
       [20, 19],
       [20, 20],
       [14, 21]], dtype=np.int8)

BFS(target_area, GOI, returnarrayD)
print("end")
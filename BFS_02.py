"""
2020/12/30
@Yuya Shimizu
@Wataru Ogurahata edit

番兵
迷路探索１（幅優先探索）
"""

import numpy as np

target_area = np.array(((9, 22), (9, 23), (9, 24), (9, 25), (10, 23), (10, 24), (10, 25)))
goi = np.array((11, 25))

mazeYedge = np.min((goi[0], np.min(target_area.T[0])))
mazeXedge = np.min((goi[1], np.min(target_area.T[1])))
mazeY1 = np.max((goi[0], np.max(target_area.T[0])))- mazeYedge + 1
mazeX1 = np.max((goi[1], np.max(target_area.T[1])))- mazeXedge + 1
maze = np.full((mazeY1, mazeX1), 9, dtype=np.int8)
route = np.zeros((mazeY1, mazeX1), dtype=np.int8)
for xy in target_area-np.array((mazeYedge, mazeXedge)): # 対象範囲を白抜き
    maze[xy[0], xy[1]] = 0
maze[goi[0]-mazeYedge, goi[1]-mazeXedge] = 1 # 流出点を設定
maze = np.vstack((np.full((mazeX1,), 9), maze, np.full((mazeX1,), 9))) # 周囲を埋め
maze = np.hstack((np.full((mazeY1+2,1), 9), maze, np.full((mazeY1+2,1), 9)))

#探索関数：ゴールしたらそのときの位置・移動数を返す
def Maze(start):
    #スタート位置（x座標, y座標, 移動回数）をセット
    pos = start

    while len(pos) > 0:#探索可能ならTrue
        x, y, depth = pos.pop(0) #リストから探索する位置を取得

        #ゴールについた時点で終了
        if maze[x][y] == 1:
            return [(x, y), depth]

        #探索済みとしてセット
        maze[x][y] = 2

        #現在位置の上下左右を探索：〇<2は壁でもなく探索済みでもないものを示す
        if maze[x-1][y] < 2:#左
            pos.append([x-1, y, depth + 1])
        if maze[x+1][y] < 2:#右
            pos.append([x+1, y, depth + 1])
        if maze[x][y-1] < 2:#上
            pos.append([x, y-1, depth + 1])
        if maze[x][y+1] < 2:#下
            pos.append([x, y+1, depth + 1])

    return False

if __name__ == '__main__':
    #迷路作成
    maze = maze.tolist()
    start = [[1, 1, 0]]     #スタート位置

    result = Maze(start)  #探索

    print(result)
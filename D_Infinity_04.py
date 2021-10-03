# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:13:46 2020

@author: mgkog
"""

import numpy as np
from math import degrees, atan, pi

a = [[10., 10.2, 10.7, 11.0, 11.3, 11.7, 11.5], 
     [2, 3, 10.1, 10.4, 11, 11.1, 11.6],
     [3, 9.1, 10, 10, 10.2, 10.3, 11],
     [3.5, 3.5, 10.4, 10.2, 12, 18, 21],
     [3, 6, 10, 13, 13.3, 14.4, 19],
     [10, 11, 10, 19, 17, 21, 23],
     [10, 10, 10, 88, 7, 90, 70]]

myarray = np.array(a)

dx = 2
dy = 2
dxy = (pow(dx,2)+pow(dy,2))**0.5
dtan = atan(dx/dy)
rows = myarray.shape[0]
cols = myarray.shape[1]
returnarrayS = np.zeros((rows, cols))
returnarrayR = np.zeros((rows, cols))


class cal():
    def DInfinity(self, arr, i, j):
        if i-1 >= 0 and j-1 >= 0 and i+1 < rows and j+1 < cols:
            b = [[0,1],[1,-1],[1,1],[2,-2],[2,1],[3,-1],[3,1],[4,-1]]
            
            s1en = (arr[i,j]-arr[i,j+1])/dx
            s2en = (arr[i,j+1]-arr[i-1,j+1])/dy
            if s1en > 0:
                ren_temp = atan(s2en/s1en)
                if ren_temp > dtan:
                    ren = dtan
                    sen = (arr[i,j]-arr[i-1,j+1])/dxy
                elif ren_temp > 0:
                    ren = ren_temp
                    sen = (pow(s1en,2)+pow(s2en,2))**0.5
                else:
                    ren = 0.00
                    sen = s1en
            else:
                ren = 0.00
                sen = s1en

            s1ne = (arr[i,j]-arr[i-1,j])/dx 
            s2ne = (arr[i-1,j]-arr[i-1,j+1])/dy
            if s1ne > 0:
                rne_temp = atan(s2ne/s1ne)
                if rne_temp > dtan:
                    rne = dtan
                    sne = (arr[i,j]-arr[i-1,j+1])/dxy
                elif rne_temp > 0:
                    rne = rne_temp
                    sne = (pow(s1ne,2)+pow(s2ne,2))**0.5
                else:
                    rne = 0.00
                    sne = s1ne
            else:
                rne = 0.00
                sne = s1ne

            s1nw = (arr[i,j]-arr[i-1,j])/dx
            s2nw = (arr[i-1,j]-arr[i-1,j-1])/dy
            if s1nw > 0:
                rnw_temp = atan(s2nw/s1nw)
                if rnw_temp > dtan:
                    rnw = dtan
                    snw = (arr[i,j]-arr[i-1,j-1])/dxy
                elif rnw_temp > 0:
                    rnw = rnw_temp
                    snw = (pow(s1nw,2)+pow(s2nw,2))**0.5
                else:
                    rnw = 0.00
                    snw = s1nw
            else:
                rnw = 0.00
                snw = s1nw

            s1wn = (arr[i,j]-arr[i,j-1])/dx
            s2wn = (arr[i,j-1]-arr[i-1,j-1])/dy
            if s1wn > 0:
                rwn_temp = atan(s2wn/s1wn)
                if rwn_temp > dtan:
                    rwn = dtan
                    swn = (arr[i,j]-arr[i-1,j-1])/dxy
                elif rwn_temp > 0:
                    rwn = rwn_temp
                    swn = (pow(s1wn,2)+pow(s2wn,2))**0.5
                else:
                    rwn = 0.00
                    swn = s1wn
            else:
                rwn = 0.00
                swn = s1wn

            s1ws = (arr[i,j]-arr[i,j-1])/dx
            s2ws = (arr[i,j-1]-arr[i+1,j-1])/dy
            if s1ws > 0:
                rws_temp = atan(s2ws/s1ws)
                if rws_temp > dtan:
                    rws = dtan
                    sws = (arr[i,j]-arr[i+1,j-1])/dxy
                elif rws_temp > 0:
                    rws = rws_temp
                    sws = (pow(s1ws,2)+pow(s2ws,2))**0.5
                else:
                    rws = 0.00
                    sws = s1ws
            else:
                rws = 0.00
                sws = s1ws

            s1sw = (arr[i,j]-arr[i+1,j])/dx
            s2sw = (arr[i+1,j]-arr[i+1,j-1])/dy
            if s1sw > 0:
                rsw_temp = atan(s2sw/s1sw)
                if rsw_temp > dtan:
                    rsw = dtan
                    ssw = (arr[i,j]-arr[i+1,j-1])/dxy
                elif rsw_temp > 0:
                    rsw = rsw_temp
                    ssw = (pow(s1sw,2)+pow(s2sw,2))**0.5
                else:
                    rsw = 0.00
                    ssw = s1sw
            else:
                rsw = 0.00
                ssw = s1sw

            s1se = (arr[i,j]-arr[i+1,j])/dx
            s2se = (arr[i+1,j]-arr[i+1,j+1])/dy
            if s1se > 0:
                rse_temp = atan(s2se/s1se)
                if rse_temp > dtan:
                    rse = dtan
                    sse = (arr[i,j]-arr[i+1,j+1])/dxy
                elif rse_temp > 0:
                    rse = rse_temp
                    sse = (pow(s1se,2)+pow(s2se,2))**0.5
                else:
                    rse = 0.00
                    sse = s1se
            else:
                rse = 0.00
                sse = s1se

            s1es = (arr[i,j]-arr[i,j+1])/dx
            s2es = (arr[i,j+1]-arr[i+1,j+1])/dy
            if s1es > 0:
                res_temp = atan(s2es/s1es)
                if res_temp > dtan:
                    res = dtan
                    ses = (arr[i,j]-arr[i+1,j+1])/dxy
                elif res_temp > 0:
                    res = rse_temp
                    ses = (pow(s1es,2)+pow(s2es,2))**0.5
                else:
                    res = 0.00
                    ses = s1es
            else:
                res = 0.00
                ses = s1es

            
            r_tuple = (ren, rne, rnw, rwn, rws, rsw, rse, res)
            s_tuple = (sen, sne, snw, swn, sws, ssw, sse, ses)
            
            smax = max(s_tuple)
            sid = s_tuple.index(smax)
            r = r_tuple[sid]
            rdeg = degrees(r)
            r1 = rdeg/45
            r2 = (45-rdeg)/45
            rg = degrees(b[sid][1]*r+b[sid][0]*pi/2)
            return smax, rg, sid, r, rdeg, r_tuple, s_tuple
            
        else:
            return np.nan, np.nan


c = cal()
for i in range(0, rows):
    for j in range(0, cols):
        args = c.DInfinity(myarray, i, j)
        returnarrayS[i][j] = args[0]
        returnarrayR[i][j] = args[1]
print(returnarrayR)

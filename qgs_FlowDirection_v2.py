#Import Modules
import numpy as np
from math import sqrt, degrees, atan, pi
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterBand,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFileDestination)
from osgeo import gdal

class FlowDirection(QgsProcessingAlgorithm):
    
    INPUT = "INPUT"
    BAND = "BAND"
    OUTPUT_S = "OUTPUT_S"
    OUTPUT_D = "OUTPUT_D"

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return FlowDirection()

    def name(self):
        return 'flowdirection'

    def displayName(self):
        return self.tr('Flow Direction')

    def group(self):
        return self.tr('KOGA Raster')

    def groupId(self):
        return 'kogaraster'

    def shortHelpString(self):
        return self.tr("No description")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT,self.tr('Input layer')))
        self.addParameter(QgsProcessingParameterBand(self.BAND,self.tr('Bamd Number'),1,self.INPUT))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT_S,self.tr('Output layer Slope'),self.tr('Tiff files (*.tif')))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT_D,self.tr('Output layer Direction'),self.tr('Tiff files (*.tif')))
    
    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        band = self.parameterAsInt(parameters, self.BAND, context)
        output_S = self.parameterAsFileOutput(parameters, self.OUTPUT_S, context)
        output_D = self.parameterAsFileOutput(parameters, self.OUTPUT_D, context)

        filename = str(layer.source())
        datasete = gdal.Open(filename, gdal.GA_ReadOnly)
        Xsize = datasete.RasterXSize
        Ysize = datasete.RasterYSize
        GT = datasete.GetGeoTransform()
        dx, dy = GT[1], GT[5]
        dxy = sqrt(pow(dx,2)+pow(dy,2))
        dtan = atan(dx/dy)
        CSR = datasete.GetProjection()
        my_band = datasete.GetRasterBand(band)

        my_array = my_band.ReadAsArray()
        returnarrayS = np.zeros((Ysize, Xsize))
        returnarrayD = np.zeros((Ysize, Xsize))

        def DInfinity(arr, i, j):
            if i-1 >= 0 and j-1 >= 0 and i+1 < Ysize and j+1 < Xsize:
                if arr[i-1][j-1] < -999 or arr[i-1][j+1] < -999 or arr[i+1][j-1] < -999 or arr[i+1][j+1] < -999:
                    return np.nan, np.nan
                else:
                    try:
                        b = [[0,1],[1,-1],[1,1],[2,-1],[2,1],[3,-1],[3,1],[4,-1]]
                        
                        s1en = (arr[i,j]-arr[i,j+1])/dx
                        s2en = (arr[i,j+1]-arr[i-1,j+1])/dy
                        if s1en > 0:
                            ren_temp = atan(s2en/s1en)
                            if ren_temp > dtan:
                                ren = dtan
                                sen = (arr[i,j]-arr[i-1,j+1])/dxy
                            elif ren_temp > 0:
                                ren = ren_temp
                                sen = sqrt(pow(s1en,2)+pow(s2en,2))
                            else:
                                ren = 0.00
                                sen = s1en
                        else:
                            s0en = (arr[i,j]-arr[i-1,j+1])/dxy
                            if  s0en > 0:
                                ren = dtan
                                sen = s0en
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
                                sne = sqrt(pow(s1ne,2)+pow(s2ne,2))
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
                                snw = sqrt(pow(s1nw,2)+pow(s2nw,2))
                            else:
                                rnw = 0.00
                                snw = s1nw
                        else:
                            s0nw = (arr[i,j]-arr[i-1,j-1])/dxy
                            if  s0nw > 0:
                                rnw = dtan
                                snw = s0nw
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
                                swn = sqrt(pow(s1wn,2)+pow(s2wn,2))
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
                                sws = sqrt(pow(s1ws,2)+pow(s2ws,2))
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
                                ssw = sqrt(pow(s1sw,2)+pow(s2sw,2))
                            else:
                                rsw = 0.00
                                ssw = s1sw
                        else:
                            s0sw = (arr[i,j]-arr[i+1,j-1])/dxy
                            if  s0sw > 0:
                                rsw = dtan
                                ssw = s0sw
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
                                sse = sqrt(pow(s1se,2)+pow(s2se,2))
                            else:
                                rse = 0.00
                                sse = s1se
                        else:
                            s0se = (arr[i,j]-arr[i+1,j+1])/dxy
                            if s0se > 0:
                                rse = dtan
                                sse = s0se
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
                                res = res_temp
                                ses = sqrt(pow(s1es,2)+pow(s2es,2))
                            else:
                                res = 0.00
                                ses = s1es
                        else:
                            s0es = (arr[i,j]-arr[i+1,j+1])/dxy
                            if  s0es > 0:
                                res = dtan
                                ses = s0es
                            else:
                                res = 0.00
                                ses = s1es
                        
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
                        return sdeg, rg, r_tuple, s_tuple
                    except:
                        return np.nan, np.nan
            else:
                return np.nan, np.nan

        #calculation
        for i in range(0, Ysize):
            for j in range(0, Xsize):
                args = DInfinity(my_array, i, j)
                returnarrayS[i][j] = args[0]
                returnarrayD[i][j] = args[1]

        dtype = gdal.GDT_Float32 #others: gdal.GDT_Byte, ...
        out_band = 1 # バンド数
        out_tiff_S = gdal.GetDriverByName('GTiff').Create(output_S, Xsize, Ysize, out_band, dtype)
        out_tiff_D = gdal.GetDriverByName('GTiff').Create(output_D, Xsize, Ysize, out_band, dtype) # 空の出力ファイル

        out_tiff_S.SetGeoTransform(GT) # 座標系指定
        out_tiff_S.SetProjection(CSR) # 空間情報を結合
        out_tiff_D.SetGeoTransform(GT)
        out_tiff_D.SetProjection(CSR)

        out_tiff_S.GetRasterBand(1).WriteArray(returnarrayS)
        out_tiff_S.FlushCache()
        out_tiff_D.GetRasterBand(1).WriteArray(returnarrayD)
        out_tiff_D.FlushCache()

        return {self.OUTPUT_S: output_S, self.OUTPUT_D: output_D}


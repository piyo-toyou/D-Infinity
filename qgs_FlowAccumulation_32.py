'''
プログラムの概要
このプログラムは、地理情報システムを処理するQGIS上において、斜面傾斜方向のデータを持つラスタを用いて、
各セルの集水面積を計算するためのプログラムである。
旧版の_31.pyと比較し、以下の点を修正した。
・それぞれのセルに対して集水面積＋１の計算を行い、セル自身が持つ集水面積を考慮した。
・最終計算結果を、集水セル数ではなく、集水面積とするために、セル数に単位セル面積を掛け合わせた
'''

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

class FlowAccumulation(QgsProcessingAlgorithm):
    
    INPUT = "INPUT"
    BAND = "BAND"
    OUTPUT_FA = "OUTPUT_FA"

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return FlowAccumulation()

    def name(self):
        return 'flowaccumulation20'

    def displayName(self):
        return self.tr('Flow Accumulation 20')

    def group(self):
        return self.tr('KOGA Raster')

    def groupId(self):
        return 'kogaraster'

    def shortHelpString(self):
        return self.tr("No description")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT,self.tr('Input layer')))
        self.addParameter(QgsProcessingParameterBand(self.BAND,self.tr('Bamd Number'),1,self.INPUT))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT_FA,self.tr('Output layer Accumulation'),self.tr('Tiff files (*.tif')))
    
    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        band = self.parameterAsInt(parameters, self.BAND, context)
        output_FA = self.parameterAsFileOutput(parameters, self.OUTPUT_FA, context)

        filename = str(layer.source())
        datasete = gdal.Open(filename, gdal.GA_ReadOnly)
        Xsize = datasete.RasterXSize
        Ysize = datasete.RasterYSize
        GT = datasete.GetGeoTransform()
        dx, dy = abs(GT[1]), abs(GT[5])
        dxy = Xsize * Ysize
        CSR = datasete.GetProjection()
        my_band = datasete.GetRasterBand(band)

        myarrayFD = my_band.ReadAsArray()
        returnarrayFA = np.full(myarrayFD.shape, -0.5, dtype=np.float64)

        def StartCellScreening(Input_FDarray, i, j): # 隣接する8セルへの流出がない「流出開始セル」と「流出中間セル」を選別し、「流出開始セル」にラスタの持つセル面積分だけ初期値を与えた
            if 135 < Input_FDarray[i,j+1] < 225 or 180 < Input_FDarray[i-1,j+1] < 270 or 225 < Input_FDarray[i-1,j] < 315 \
                    or 270 < Input_FDarray[i-1,j-1] or 315 < Input_FDarray[i,j-1] or 0 <= Input_FDarray[i,j-1] < 45 or 0 < Input_FDarray[i+1,j-1] < 90 \
                    or 45 < Input_FDarray[i+1,j] < 135 or 90 < Input_FDarray[i+1,j+1] < 180: 
                return -1.0
            else:
                return 1

        def Accumulation(Input_FDarray, Output_FAarray, i, j):
            Accumulation_Value = 0 # 周囲の8セルからの集水セル数を初期化
            # east
            if Input_FDarray[i,j+1] <= 135 or Input_FDarray[i,j+1] >= 225:
                pass
            elif 135 < Input_FDarray[i,j+1] < 180 and Output_FAarray[i,j+1] > 0:
                Accumulation_Value += (Output_FAarray[i,j+1])*((Input_FDarray[i,j+1]-135)/45)
            elif 180 <= Input_FDarray[i,j+1] < 225 and Output_FAarray[i,j+1] > 0:
                Accumulation_Value += (Output_FAarray[i,j+1])*((225-Input_FDarray[i,j+1])/45)
            elif Input_FDarray[i,j+1] == -1:
                pass
            else:
                return False
            
            #north - east
            if Input_FDarray[i-1,j+1] <= 180 or Input_FDarray[i-1,j+1] >= 270:
                pass
            elif 180 < Input_FDarray[i-1,j+1] < 225 and Output_FAarray[i-1,j+1] > 0:
                Accumulation_Value += (Output_FAarray[i-1,j+1])*((Input_FDarray[i-1,j+1]-180)/45)
            elif 225 <= Input_FDarray[i-1,j+1] < 270 and Output_FAarray[i-1,j+1] > 0:
                Accumulation_Value += (Output_FAarray[i-1,j+1])*((270-Input_FDarray[i-1,j+1])/45)
            elif Input_FDarray[i-1,j+1] == -1:
                pass
            else:
                return False
            
            # north
            if Input_FDarray[i-1,j] <= 225 or Input_FDarray[i-1,j] >= 315:
                pass
            elif 225 < Input_FDarray[i-1,j] < 270 and Output_FAarray[i-1,j] > 0:
                Accumulation_Value += (Output_FAarray[i-1,j])*((Input_FDarray[i-1,j]-225)/45)
            elif 270 <= Input_FDarray[i-1,j] < 315 and Output_FAarray[i-1,j] > 0:
                Accumulation_Value += (Output_FAarray[i-1,j])*((315-Input_FDarray[i-1,j])/45)
            elif Input_FDarray[i-1,j] == -1:
                pass
            else:
                return False

            # north - west
            if Input_FDarray[i-1,j-1] <= 270:
                pass
            elif 270 < Input_FDarray[i-1,j-1] < 315 and Output_FAarray[i-1,j-1] > 0:
                Accumulation_Value += (Output_FAarray[i-1,j-1])*((Input_FDarray[i-1,j-1]-270)/45)
            elif 315 <= Input_FDarray[i-1,j-1] and Output_FAarray[i-1,j-1] > 0:
                Accumulation_Value += (Output_FAarray[i-1,j-1])*((360-Input_FDarray[i-1,j-1])/45)
            elif Input_FDarray[i-1,j-1] == -1:
                pass
            else:
                return False

            # west
            if Input_FDarray[i,j-1] >= 45 and Input_FDarray[i,j-1] <= 315:
                pass
            elif  0 <= Input_FDarray[i,j-1] < 45 and Output_FAarray[i,j-1] > 0:
                Accumulation_Value += (Output_FAarray[i,j-1])*(1-(Input_FDarray[i,j-1]/45))
            elif 315 < Input_FDarray[i,j-1] and Output_FAarray[i,j-1] > 0:
                Accumulation_Value += (Output_FAarray[i,j-1])*((Input_FDarray[i,j-1]-315)/45)
            elif Input_FDarray[i,j-1] == -1:
                pass
            else:
                return False

            # south - west
            if Input_FDarray[i+1,j-1] >= 90 or Input_FDarray[i+1,j-1] <= 0:
                pass
            elif Input_FDarray[i+1,j-1] < 45 and Output_FAarray[i+1,j-1] > 0:
                Accumulation_Value += (Output_FAarray[i+1,j-1])*(Input_FDarray[i+1,j-1]/45)
            elif 45 <= Input_FDarray[i+1,j-1] < 90 and Output_FAarray[i+1,j-1] > 0:
                Accumulation_Value += (Output_FAarray[i+1,j-1])*((90-Input_FDarray[i+1,j-1])/45)
            elif Input_FDarray[i+1,j-1] == -1:
                pass
            else:
                return False
            
            # south
            if Input_FDarray[i+1,j] <= 45 or Input_FDarray[i+1,j] >= 135:
                pass
            elif 45 < Input_FDarray[i+1,j] < 90 and Output_FAarray[i+1,j] > 0:
                Accumulation_Value += (Output_FAarray[i+1,j])*((Input_FDarray[i+1,j]-45)/45)
            elif 90 <= Input_FDarray[i+1,j] < 135 and Output_FAarray[i+1,j] > 0:
                Accumulation_Value += (Output_FAarray[i+1,j])*((135-Input_FDarray[i+1,j])/45)
            elif Input_FDarray[i+1,j] == -1:
                pass
            else:
                return False

            # south - east
            if Input_FDarray[i+1,j+1] <= 90 or Input_FDarray[i+1,j+1] >= 180:
                pass
            elif 90 < Input_FDarray[i+1,j+1] < 135 and Output_FAarray[i+1,j+1] > 0:
                Accumulation_Value += (Output_FAarray[i+1,j+1])*((Input_FDarray[i+1,j+1]-90)/45)
            elif 135 <= Input_FDarray[i+1,j+1] < 180 and Output_FAarray[i+1,j+1] > 0:
                Accumulation_Value += (Output_FAarray[i+1,j+1])*((180-Input_FDarray[i+1,j+1])/45)
            elif Input_FDarray[i+1,j+1] == -1:
                pass
            else:
                return False

            return Accumulation_Value + 1 # 最後にセル自身がもつ面積（セル数1）を足し合わせる

        #Calculation
        #Start Cell Screaning 隣接する8セルへの流出がない「流出開始セル」と「流出中間セル」を選別        
        for i, j in np.argwhere(myarrayFD != -1):
            returnarrayFA[i][j] = StartCellScreening(myarrayFD, i, j)

        #Flow Accumulation 隣接する8セルから、集水セル数を計算
        loop = 0
        while np.argwhere(returnarrayFA == -1).size:
            for i, j in np.argwhere(returnarrayFA == -1):
                f = Accumulation(myarrayFD, returnarrayFA, i, j)
                if f:
                    returnarrayFA[i][j] = f
                else:
                    pass
            loop += 1
            if loop == 10000:
                break


        # 最後にセル数に単位セルサイズを掛け合わせて集水面積を計算
        returnarrayFArea = returnarrayFA * dxy

        dtype = gdal.GDT_Float32 #others: gdal.GDT_Byte, ...
        out_band = 1 # バンド数
        out_tiff_FA = gdal.GetDriverByName('GTiff').Create(output_FA, Xsize, Ysize, out_band, dtype)

        out_tiff_FA.SetGeoTransform(GT) # 座標系指定
        out_tiff_FA.SetProjection(CSR) # 空間情報を結合

        out_tiff_FA.GetRasterBand(1).WriteArray(returnarrayFArea)
        out_tiff_FA.FlushCache()

        return {self.OUTPUT_FA: output_FA}


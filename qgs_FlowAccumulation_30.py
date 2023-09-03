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
        CSR = datasete.GetProjection()
        my_band = datasete.GetRasterBand(band)

        myarrayFD = my_band.ReadAsArray()
        returnarrayFA = np.full(myarrayFD.shape, -0.5)

        def Zero(arr, i, j):
            if 135 < arr[i,j+1] < 225 or 180 < arr[i-1,j+1] < 270 or 225 < arr[i-1,j] < 315 \
                    or 270 < arr[i-1,j-1] or 315 < arr[i,j-1] or 0 <= arr[i,j-1] < 45 or 0 < arr[i+1,j-1] < 90 \
                    or 45 < arr[i+1,j] < 135 or 90 < arr[i+1,j+1] < 180: 
                return -1.0
            else:
                return 0

        def ACM(arr1, arr2, i, j):
            d = 0
            # east
            if arr1[i,j+1] <= 135 or arr1[i,j+1] >= 225:
                pass
            elif 135 < arr1[i,j+1] < 180 and arr2[i,j+1] > 0:
                d += (arr2[i,j+1]+1)*((arr1[i,j+1]-135)/45)
            elif 135 < arr1[i,j+1] < 180 and arr2[i,j+1] == 0:
                d += (arr1[i,j+1]-135)/45
            elif 180 <= arr1[i,j+1] < 225 and arr2[i,j+1] > 0:
                d += (arr2[i,j+1]+1)*((225-arr1[i,j+1])/45)
            elif 180 <= arr1[i,j+1] < 225 and arr2[i,j+1] == 0:
                d += (225-arr1[i,j+1])/45
            elif arr1[i,j+1] == -1:
                pass
            else:
                return False
            
            #north - east
            if arr1[i-1,j+1] <= 180 or arr1[i-1,j+1] >= 270:
                pass
            elif 180 < arr1[i-1,j+1] < 225 and arr2[i-1,j+1] > 0:
                d += (arr2[i-1,j+1]+1)*((arr1[i-1,j+1]-180)/45)
            elif 180 < arr1[i-1,j+1] < 225 and arr2[i-1,j+1] == 0:
                d += (arr1[i-1,j+1]-180)/45
            elif 225 <= arr1[i-1,j+1] < 270 and arr2[i-1,j+1] > 0:
                d += (arr2[i-1,j+1]+1)*((270-arr1[i-1,j+1])/45)
            elif 225 <= arr1[i-1,j+1] < 270 and arr2[i-1,j+1] == 0:
                d += (270-arr1[i-1,j+1])/45
            elif arr1[i-1,j+1] == -1:
                pass
            else:
                return False
            
            # north
            if arr1[i-1,j] <= 225 or arr1[i-1,j] >= 315:
                pass
            elif 225 < arr1[i-1,j] < 270 and arr2[i-1,j] > 0:
                d += (arr2[i-1,j]+1)*((arr1[i-1,j]-225)/45)
            elif 225 < arr1[i-1,j] < 270 and arr2[i-1,j] == 0:
                d += (arr1[i-1,j]-225)/45
            elif 270 <= arr1[i-1,j] < 315 and arr2[i-1,j] > 0:
                d += (arr2[i-1,j]+1)*((315-arr1[i-1,j])/45)
            elif 270 <= arr1[i-1,j] < 315 and arr2[i-1,j] == 0:
                d += (315-arr1[i-1,j])/45
            elif arr1[i-1,j] == -1:
                pass
            else:
                return False

            # north - west
            if arr1[i-1,j-1] <= 270:
                pass
            elif 270 < arr1[i-1,j-1] < 315 and arr2[i-1,j-1] > 0:
                d += (arr2[i-1,j-1]+1)*((arr1[i-1,j-1]-270)/45)
            elif 270 < arr1[i-1,j-1] < 315 and arr2[i-1,j-1] == 0:
                d += (arr1[i-1,j-1]-270)/45
            elif 315 <= arr1[i-1,j-1] and arr2[i-1,j-1] > 0:
                d += (arr2[i-1,j-1]+1)*((360-arr1[i-1,j-1])/45)
            elif 315 <= arr1[i-1,j-1] and arr2[i-1,j-1] == 0:
                d += (360-arr1[i-1,j-1])/45
            elif arr1[i-1,j-1] == -1:
                pass
            else:
                return False

            # west
            if arr1[i,j-1] >= 45 and arr1[i,j-1] <= 315:
                pass
            elif  0 <= arr1[i,j-1] < 45 and arr2[i,j-1] > 0:
                d += (arr2[i,j-1]+1)*(1-(arr1[i,j-1]/45))
            elif  0 <= arr1[i,j-1] < 45 and arr2[i,j-1] == 0:
                d += 1-(arr1[i,j-1]/45)
            elif 315 < arr1[i,j-1] and arr2[i,j-1] > 0:
                d += (arr2[i,j-1]+1)*((arr1[i,j-1]-315)/45)
            elif 315 < arr1[i,j-1] and arr2[i,j-1] == 0:
                d += (arr1[i,j-1]-315)/45
            elif arr1[i,j-1] == -1:
                pass
            else:
                return False

            # south - west
            if arr1[i+1,j-1] >= 90 or arr1[i+1,j-1] <= 0:
                pass
            elif arr1[i+1,j-1] < 45 and arr2[i+1,j-1] > 0:
                d += (arr2[i+1,j-1]+1)*(arr1[i+1,j-1]/45)
            elif arr1[i+1,j-1] < 45 and arr2[i+1,j-1] == 0:
                d += arr1[i+1,j-1]/45
            elif 45 <= arr1[i+1,j-1] < 90 and arr2[i+1,j-1] > 0:
                d += (arr2[i+1,j-1]+1)*((90-arr1[i+1,j-1])/45)
            elif 45 <= arr1[i+1,j-1] < 90 and arr2[i+1,j-1] == 0:
                d += (90-arr1[i+1,j-1])/45
            elif arr1[i+1,j-1] == -1:
                pass
            else:
                return False
            
            # south
            if arr1[i+1,j] <= 45 or arr1[i+1,j] >= 135:
                pass
            elif 45 < arr1[i+1,j] < 90 and arr2[i+1,j] > 0:
                d += (arr2[i+1,j]+1)*((arr1[i+1,j]-45)/45)
            elif 45 < arr1[i+1,j] < 90 and arr2[i+1,j] == 0:
                d += (arr1[i+1,j]-45)/45
            elif 90 <= arr1[i+1,j] < 135 and arr2[i+1,j] > 0:
                d += (arr2[i+1,j]+1)*((135-arr1[i+1,j])/45)
            elif 90 <= arr1[i+1,j] < 135 and arr2[i+1,j] == 0:
                d += (135-arr1[i+1,j])/45
            elif arr1[i+1,j] == -1:
                pass
            else:
                return False

            # south - east
            if arr1[i+1,j+1] <= 90 or arr1[i+1,j+1] >= 180:
                pass
            elif 90 < arr1[i+1,j+1] < 135 and arr2[i+1,j+1] > 0:
                d += (arr2[i+1,j+1]+1)*((arr1[i+1,j+1]-90)/45)
            elif 90 < arr1[i+1,j+1] < 135 and arr2[i+1,j+1] == 0:
                d += (arr1[i+1,j+1]-90)/45
            elif 135 <= arr1[i+1,j+1] < 180 and arr2[i+1,j+1] > 0:
                d += (arr2[i+1,j+1]+1)*((180-arr1[i+1,j+1])/45)
            elif 135 <= arr1[i+1,j+1] < 180 and arr2[i+1,j+1] == 0:
                d += (180-arr1[i+1,j+1])/45
            elif arr1[i+1,j+1] == -1:
                pass
            else:
                return False

            return d

        #Calculation
        #Zero        
        for i, j in np.argwhere(myarrayFD != -1):
            returnarrayFA[i][j] = Zero(myarrayFD, i, j)

        #Flow Accumration (?)

        loop = 0
        while np.argwhere(returnarrayFA == -1).size:
            for i, j in np.argwhere(returnarrayFA == -1):
                f = ACM(myarrayFD, returnarrayFA, i, j)
                if f:
                    returnarrayFA[i][j] = f
                else:
                    pass
            loop += 1
            if loop == 10000:
                break

        dtype = gdal.GDT_Float32 #others: gdal.GDT_Byte, ...
        out_band = 1 # バンド数
        out_tiff_FA = gdal.GetDriverByName('GTiff').Create(output_FA, Xsize, Ysize, out_band, dtype)

        out_tiff_FA.SetGeoTransform(GT) # 座標系指定
        out_tiff_FA.SetProjection(CSR) # 空間情報を結合

        out_tiff_FA.GetRasterBand(1).WriteArray(returnarrayFA)
        out_tiff_FA.FlushCache()

        return {self.OUTPUT_FA: output_FA}


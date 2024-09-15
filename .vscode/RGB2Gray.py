import sys
import cv2
import numpy as np

gamma22LUT = np.array([pow(x/255.0 , 2.2) for x in range(256)],
                         dtype='float32')
infile, outfile = sys.argv[1], sys.argv[2]
img_bgr = cv2.imread(infile)
img_bgrL = cv2.LUT(img_bgr, gamma22LUT)
img_grayL = cv2.cvtColor(img_bgrL, cv2.COLOR_BGR2GRAY)
img_gray = pow(img_grayL, 1.0/2.2) * 255
cv2.imwrite(outfile, img_gray)

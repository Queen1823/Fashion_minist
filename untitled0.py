# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:36:37 2021

@author: Administrator
"""

import cv2
import numpy as np
img = cv2.imread('image0.jpg', 1)
cv2.imshow("image", img)
height, width = img.shape[:2]  # 405x413
# 在原图像和目标图像上各选择三个点
matSrc = np.float32([[0, 0],[0, height-1],[width-1, 0]])
matDst = np.float32([[0, 0],[30, height-30],[width-30, 30]])
# 得到变换矩阵
matAffine = cv2.getAffineTransform(matSrc, matDst)
# 进行仿射变换
dst = cv2.warpAffine(img, matAffine, (width,height))
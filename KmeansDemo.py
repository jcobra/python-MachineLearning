# -*- coding: utf-8 -*-
"""
K-means实现图像分割demo
"""
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as image
from sklearn.cluster import KMeans


def loadData(filePath):
    f = open(filePath, 'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0, y/256.0, z/256.0])
    f.close()
    return np.mat(data),m,n


def main():
    imgData, row, col = loadData("data/bull.jpg")
    km = KMeans(n_clusters=3) # 聚类中心数为3
    label = km.fit_predict(imgData)
    label = label.reshape([row, col])
    # print(label)
    pic_new = image.new("L", (row,col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), int(256 / (label[i][j]+1)))
    pic_new.save("data/result-bull-4.jpg", "JPEG")


if __name__ == '__main__':
    main()








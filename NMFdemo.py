# -*- coding: utf-8 -*-
"""
降维NMF算法demo，Olivetti人脸数据特征提取
"""
import matplotlib.pyplot as plt
# 加载法包
from sklearn import decomposition
# 人脸数据集
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState


def plot_gallery(title, images, n_col, n_row):
    # 设置人脸数据大小
    image_shape = (64, 64)
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i+1)
        vmax = max(comp.max(), -comp.min())
        # 对数值归一化，并以灰度图形式显示
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        # 去除子图的坐标轴标签
        plt.xticks(())
        plt.yticks(())
    # 对子图位置及间隔调整
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)


def main():
    n_row, n_col = 2, 3
    n_components = n_row * n_col
    dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
    faces = dataset.data
    plot_gallery("First centered Olivetti faces", faces[:n_components], n_col, n_row)

    estimators = [
        ('Eigenfaces - PCA use randomized SVD ', decomposition.PCA(n_components=6, whiten=True)),
        ('Non-negative components - NMF', decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))
    ]

    for name, estimator in estimators:
        print("Extracting the top %d %s ..." % (n_components, name))
        print(faces.shape)
        # 分别调用PCA和NMF提取特征
        estimator.fit(faces)
        # 获取特征
        components_ = estimator.components_
        # 按照固定的格式进行排列
        plot_gallery(name, components_[:n_components], n_col, n_row)
    plt.show()


if __name__ == '__main__':
    main()

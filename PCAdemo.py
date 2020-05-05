# -*- coding: utf-8 -*-
"""
降维PCA算法demo，鸢尾花数据降维展示
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


def main():
    data = load_iris()
    # 使用y表示数据集中的标签
    y = data.target
    # 使用X表示数据集中的属性数据
    X = data.data
    # 加载PCA算法，设置降维后主成分数目为2
    pca = PCA(n_components=2)
    # 对原始数据降维，保存在reduce_X中
    reduce_X = pca.fit_transform(X)

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduce_X)):
        if y[i] == 0:
            red_x.append(reduce_X[i][0])
            red_y.append(reduce_X[i][1])
        elif y[i] == 1:
            blue_x.append(reduce_X[i][0])
            blue_y.append(reduce_X[i][1])
        else:
            green_x.append(reduce_X[i][0])
            green_y.append(reduce_X[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()


if __name__ == '__main__':
    main()




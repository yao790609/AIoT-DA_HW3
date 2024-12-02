# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:47:27 2024

@author: yao79
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D

# 產生 600 個隨機點
np.random.seed(0)  # 使得每次隨機數相同
mean = [0, 0]  # 中心在 (0, 0)
covariance = [[10, 0], [0, 10]]  # 以方差為 10 的高斯分佈
X_2d = np.random.multivariate_normal(mean, covariance, 600)

# 定義高斯函數
def gaussian_function(x1, x2):
    return np.exp(-(x1**2 + x2**2) / 20)

# 定義 Streamlit 用戶界面
st.title('SVM 分類與距離閾值調整')

# 拉桿選擇距離閾值
distance_threshold = st.slider('選擇分類距離閾值', min_value=0.0, max_value=10.0, value=4.0)

# 計算每個點到原點的距離
distances = np.linalg.norm(X_2d, axis=1)

# 根據距離，將這些點分成兩類
Y = np.where(distances < distance_threshold, 0, 1)

# 計算 x3，並將 x1, x2, x3 合併成一個三維特徵矩陣 X
x3 = np.array([gaussian_function(x1, x2) for x1, x2 in X_2d])
X_3d = np.column_stack((X_2d, x3))

# 使用 LinearSVC 訓練模型
model = LinearSVC()
model.fit(X_3d, Y)

# 獲取模型係數和截距
coef = model.coef_
intercept = model.intercept_

# 創建 3D 散點圖
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 顯示兩類點（0 和 1）
ax.scatter(X_3d[Y == 0, 0], X_3d[Y == 0, 1], X_3d[Y == 0, 2], color='r', label='Class 0')
ax.scatter(X_3d[Y == 1, 0], X_3d[Y == 1, 1], X_3d[Y == 1, 2], color='b', label='Class 1')

# 繪製超平面
xx, yy = np.meshgrid(np.linspace(-10, 10, 30), np.linspace(-10, 10, 30))
zz = (-coef[0, 0] * xx - coef[0, 1] * yy - intercept) / coef[0, 2]

# 繪製灰色超平面
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

# 設置標籤
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title(f'3D Scatter Plot with SVM Hyperplane (Threshold: {distance_threshold})')

# 顯示圖例
ax.legend()

# 顯示圖形
st.pyplot(fig)

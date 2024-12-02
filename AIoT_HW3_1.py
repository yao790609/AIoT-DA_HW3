# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:58:44 2024

@author: yao79
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 設定隨機種子，確保結果可重複
np.random.seed(42)

# 1. 產生300個0~1000範圍的亂數值
X = np.random.uniform(0, 1000, size=(300, 1))

# 2. 設定分類標籤
y = np.where((X >= 500) & (X <= 800), 1, 0).ravel()

# 資料標準化（對RBF核心很重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 將資料分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# 4. SVM with RBF Kernel
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# 5. 繪製圖表
plt.figure(figsize=(15, 5))

# Logistic Regression 圖
plt.subplot(121)
plt.scatter(X[y == 0], np.zeros(len(X[y == 0])), c='blue', label='Class 0')
plt.scatter(X[y == 1], np.ones(len(X[y == 1])), c='red', label='Class 1')

# 繪製Logistic Regression決策邊界
X_plot = np.linspace(0, 1000, 100).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_plot = lr_model.predict_proba(X_plot_scaled)[:, 1]
plt.plot(X_plot, y_plot, color='green', label='Logistic Regression Decision Boundary')
plt.title('Logistic Regression')
plt.xlabel('X值')
plt.ylabel('預測機率')
plt.legend()
plt.show()

# SVM 圖
plt.subplot(122)
plt.scatter(X[y == 0], np.zeros(len(X[y == 0])), c='blue', label='Class 0')
plt.scatter(X[y == 1], np.ones(len(X[y == 1])), c='red', label='Class 1')

# 繪製RBF核心SVM的決策邊界（概率）
X_plot = np.linspace(0, 1000, 100).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_plot_prob = svm_model.predict_proba(X_plot_scaled)[:, 1]
plt.plot(X_plot, y_plot_prob, color='green', label='SVM RBF Decision Boundary')
plt.title('Support Vector Machine (RBF Kernel)')
plt.xlabel('X值')
plt.ylabel('預測機率')
plt.legend()

plt.tight_layout()
plt.show()

# 輸出模型準確度
print(f"Logistic Regression 準確度: {lr_model.score(X_test, y_test)*100:.2f}%")
print(f"SVM (RBF Kernel) 準確度: {svm_model.score(X_test, y_test)*100:.2f}%")
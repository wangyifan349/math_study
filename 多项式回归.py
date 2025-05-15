import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. 生成数据 (真实函数 y = 1 + 2x + 3x^2 + 噪声)
np.random.seed(0)  # 固定随机种子，保证结果可复现
X = 2 - 3 * np.random.rand(100, 1)  # 生成100个随机样本，分布在[-1, 2]区间
y = 1 + 2 * X + 3 * X**2 + np.random.randn(100, 1)  # 目标值，含正态噪声

# 2. 划分训练集和测试集，80%训练，20%测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. 训练简单线性回归模型（不做多项式扩展）
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)  # 训练模型
y_train_pred_lin = lin_reg.predict(X_train)  # 训练集预测值
y_test_pred_lin = lin_reg.predict(X_test)    # 测试集预测值

# 4. 通过循环训练不同阶数的多项式回归模型，记录指标
max_degree = 5  # 最大多项式阶数
train_mse = []  # 训练集均方误差列表
test_mse = []   # 测试集均方误差列表
train_r2 = []   # 训练集R²得分列表
test_r2 = []    # 测试集R²得分列表

for degree in range(1, max_degree + 1):
    # 4.1 多项式特征转换，比如degree=2，将x转成[x, x^2]
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)  # 训练集特征转换
    X_test_poly = poly_features.transform(X_test)        # 测试集特征转换（使用训练集的变换）

    # 4.2 训练线性回归模型（多项式特征下）
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)

    # 4.3 获得训练和测试的预测结果
    y_train_pred = poly_reg.predict(X_train_poly)
    y_test_pred = poly_reg.predict(X_test_poly)

    # 4.4 计算训练和测试的均方误差 MSE
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))

    # 4.5 计算训练和测试R²得分（越接近1越好）
    train_r2.append(r2_score(y_train, y_train_pred))
    test_r2.append(r2_score(y_test, y_test_pred))

    # 4.6 打印当前阶数的指标结果
    print(f"Degree {degree}: Train MSE={train_mse[-1]:.4f}, Test MSE={test_mse[-1]:.4f}, "
          f"Train R2={train_r2[-1]:.4f}, Test R2={test_r2[-1]:.4f}")

# 5. 绘制训练和测试的均方误差随多项式阶数变化曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_degree + 1), train_mse, marker='o', label='Train MSE')
plt.plot(range(1, max_degree + 1), test_mse, marker='o', label='Test MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('训练集与测试集 MSE 随多项式阶数变化')
plt.legend()
plt.grid(True)
plt.show()

# 6. 绘制训练和测试的R²得分随多项式阶数变化曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_degree + 1), train_r2, marker='o', label='Train R²')
plt.plot(range(1, max_degree + 1), test_r2, marker='o', label='Test R²')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('训练集与测试集 R² 随多项式阶数变化')
plt.legend()
plt.grid(True)
plt.show()

# 7. 选择测试误差最低的多项式阶数作为“最佳”模型
best_degree = np.argmin(test_mse) + 1  # argmin返回索引，degree从1开始所以+1
print(f"测试集MSE表现最佳的多项式阶数是：{best_degree}")

# 8. 使用最佳阶数训练最终模型（使用全数据）
poly_features = PolynomialFeatures(degree=best_degree, include_bias=False)
X_all_poly = poly_features.fit_transform(X)
poly_reg_best = LinearRegression()
poly_reg_best.fit(X_all_poly, y)

# 9. 生成更多未来可预测的点（X轴范围扩大）用于绘制平滑曲线
X_future = np.linspace(X.min() - 1, X.max() + 1, 200).reshape(-1, 1)
X_future_poly = poly_features.transform(X_future)
y_future_pred = poly_reg_best.predict(X_future_poly)

# 10. 可视化最终拟合结果和未来预测
plt.scatter(X, y, color='blue', label='原始训练数据')
plt.plot(X_future, y_future_pred, color='orange', linewidth=2,
         label=f'最佳多项式拟合 (degree={best_degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('多项式回归拟合与未来预测')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

print("欢迎使用数据分析工具!")
print("此工具支持线性回归和多项式回归，并计算数据统计指标（中位数、均值、标准差），用于金融预测。")
print("请输入数据:")

x_input = input("请输入自变量 X 数据（例如: 1,2,3,4,5）：")
x_parts = x_input.split(',')
temp_list = []
i = 0
while i < len(x_parts):
    part = x_parts[i].strip()
    if part != "":
        try:
            num = float(part)
            temp_list.append(num)
        except Exception as e:
            print("X 数据解析错误, 请检查输入格式。")
            exit(1)
    i = i + 1
X_data = np.array(temp_list)

y_input = input("请输入因变量 Y 数据（例如: 2,4,6,8,10）：")
y_parts = y_input.split(',')
temp_list2 = []
j = 0
while j < len(y_parts):
    part = y_parts[j].strip()
    if part != "":
        try:
            num = float(part)
            temp_list2.append(num)
        except Exception as e:
            print("Y 数据解析错误, 请检查输入格式。")
            exit(1)
    j = j + 1
y_data = np.array(temp_list2)

if X_data.shape[0] != y_data.shape[0]:
    print("X 与 Y 数据数量不同。")
    exit(1)

X_data = X_data.reshape(-1, 1)

# 计算数据统计指标
X_median = np.median(X_data)
X_mean = np.mean(X_data)
X_std = np.std(X_data)
y_median = np.median(y_data)
y_mean = np.mean(y_data)
y_std = np.std(y_data)

print("X 数据统计指标：")
print("中位数:", X_median)
print("均值:", X_mean)
print("标准差:", X_std)
print("Y 数据统计指标：")
print("中位数:", y_median)
print("均值:", y_mean)
print("标准差:", y_std)

print("请选择回归模型：")
print("1. 线性回归")
print("2. 多项式回归（2阶多项式）")
model_choice = input("请输入数字（1 或 2）：").strip()

if model_choice == "1":
    model = LinearRegression()
    model_name = "线性回归"
elif model_choice == "2":
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])
    model_name = "2阶多项式回归"
else:
    print("无效的选择。")
    exit(1)

model.fit(X_data, y_data)
print(model_name + " 模型拟合完成。")
print("模型参数：")
if model_choice == "1":
    print("截距:", model.intercept_)
    print("斜率:", model.coef_)
else:
    linear_model = model.named_steps['linear']
    print("截距:", linear_model.intercept_)
    print("系数:", linear_model.coef_)

x_min = X_data.min()
x_max = X_data.max()
x_plot = np.linspace(x_min, x_max, 100)
x_plot = x_plot.reshape(-1, 1)
y_plot = model.predict(x_plot)

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.scatter(X_data, y_data, color="blue", label="数据点")
plt.plot(x_plot, y_plot, color="red", label="拟合曲线")
plt.title(model_name + " 拟合结果")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.subplot(2, 1, 2)
# 显示 X, Y 均值、中位数、标准差信息的柱状图
labels = ['X均值', 'X中位数', 'X标准差', 'Y均值', 'Y中位数', 'Y标准差']
values = [X_mean, X_median, X_std, y_mean, y_median, y_std]
index = range(len(labels))
bars = plt.bar(index, values, color='gray')
plt.xticks(index, labels)
plt.title("基本统计指标")
plt.tight_layout()
plt.show()

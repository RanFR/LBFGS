import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义x和y的数据范围
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

# 将x和y转换为网格形式
X, Y = np.meshgrid(x, y)

# 计算函数值
Z = (1 - X)**2 + 100 * (Y - X**2)**2

# 创建一个新的图形
fig = plt.figure(figsize=(10, 8))

# 添加一个3D子图
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 添加颜色条
fig.colorbar(surf)

# 设置坐标轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 设置标题
ax.set_title(r'3D Surface of $(1-x)^2 + 100(y-x^2)^2$')

# 显示图形
plt.show()

import sympy as sp

# 定义符号变量
x, y, z, t, s = sp.symbols('x y z t s')
# ================================
# 1. 微分计算
# ================================
# 基本微分
expr = sp.sin(x) * sp.exp(x)
diff_expr = sp.diff(expr, x)
print(f"一阶导数: d/dx({expr}) = {diff_expr}")
# 二阶导数
second_derivative = sp.diff(expr, x, 2)
print(f"二阶导数: d²/dx²({expr}) = {second_derivative}")
# ================================
# 2. 偏微分
# ================================
# 偏微分计算
expr2 = x**2 * y**3 * z**4
partial_diff_expr = sp.diff(expr2, y, 2)
print(f"对y的二阶偏导数: ∂²/∂y²({expr2}) = {partial_diff_expr}")
# ================================
# 3. 常微分方程
# ================================
# 齐次线性微分方程求解
f = sp.Function('f')
deqn = sp.Eq(f(x).diff(x) + f(x), 0)
solution_homo = sp.dsolve(deqn, f(x))
print(f"齐次线性微分方程的解: {solution_homo}")
# 非齐次微分方程求解
deqn2 = sp.Eq(f(x).diff(x, x) + f(x), sp.cos(x))
solution_non_homo = sp.dsolve(deqn2, f(x))
print(f"非齐次微分方程的解: {solution_non_homo}")
# ================================
# 4. 积分计算
# ================================
# 定积分
integral_expr = sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))
print(f"定积分: ∫exp(-x^2) dx from -∞ to ∞ = {integral_expr}")
# 不定积分
indefinite_integral_expr = sp.integrate(sp.sin(x) * sp.cos(x), x)
print(f"不定积分: ∫sin(x)cos(x) dx = {indefinite_integral_expr}")
# 曲线积分
r = sp.Matrix([sp.cos(t), sp.sin(t)])  # 参数化单位圆
expr3 = x**2 + y**2
curve_integral = sp.integrate(expr3.subs({x: r[0], y: r[#citation-1](citation-1)}) * r.norm().diff(t), (t, 0, 2*sp.pi))
print(f"曲线积分: ∫(x^2 + y^2) ds over C = {curve_integral}")
# ================================
# 5. 线性代数
# ================================
# 矩阵定义
A = sp.Matrix([[2, 1], [1, 2]])
# 特征值和特征向量
eigenvals = A.eigenvals()
eigenvects = A.eigenvects()
print(f"矩阵的特征值: {eigenvals}")
print(f"矩阵的特征向量: {eigenvects}")
# 行列式
determinant = A.det()
print(f"矩阵的行列式: {determinant}")
# LU分解
P, L, U = A.LUdecomposition()
print(f"LU 分解结果: L = {L}, U = {U}")
# 拉普拉斯展开式计算行列式
laplace_determinant = A.det(method='laplace')
print(f"通过拉普拉斯展开式计算行列式: {laplace_determinant}")
# ================================
# 6. 拉普拉斯变换和傅里叶变换
# ================================
# 拉普拉斯变换
laplace_transform = sp.laplace_transform(sp.sin(x), x, s)
print(f"拉普拉斯变换: L{sp.sin(x)} = {laplace_transform[0]}")
# 傅里叶变换
fourier_transform = sp.fourier_transform(sp.exp(-x**2), x, s)
print(f"傅里叶变换: F{sp.exp(-x**2)} = {fourier_transform}")
# 显示结果
print("综合数学计算已完成。")





import sympy as sp
def solve_linear_ode_step_by_step(eq, function):
    # 提取线性微分方程的一般形式 M(x)y' + N(x)y = P(x)
    x = eq.lhs.args[0].args[0]
    y = function(x)
    # 将方程转换为标准形式： y' + a(x)y = b(x)
    a = sp.simplify(eq.lhs.coeff(y)/eq.lhs.coeff(y.diff(x)))
    b = sp.simplify(eq.rhs/eq.lhs.coeff(y.diff(x)))
    # 输出标准化的方程
    print(f"将方程标准化为: y' + ({a})y = {b}")
    # 计算积分因子 μ(x) = exp(integral(a(x)dx))
    integrating_factor = sp.exp(sp.integrate(a, x))
    print(f"积分因子 μ(x) = exp(∫{a} dx) = {integrating_factor}")
    # 通过积分因子法重写方程得 d/dx(μ(x)y) = μ(x)b(x)
    left_side = integrating_factor * y
    right_side = integrating_factor * b
    print(f"重写方程为: d/dx({left_side}) = {right_side}")
    # 计算两侧定积分得 μ(x)y = ∫(μ(x)b(x)) dx + C
    y_solution = sp.integrate(right_side, x) / integrating_factor
    general_solution = y_solution + (1/integrating_factor) * sp.symbols('C')
    print(f"通过积分得到: y = {y_solution} + C/μ(x)")
    return sp.simplify(general_solution)
# 定义变量和函数
x = sp.symbols('x')
y = sp.Function('y')
# 测试用例
eq = sp.Eq(y(x).diff(x) + y(x), x)
print("例题: y' + y = x")
solution = solve_linear_ode_step_by_step(eq, y)
print(f"解为: {solution}\n=========================================\n")
eq2 = sp.Eq(y(x).diff(x) - 2*y(x), sp.exp(x))
print("例题: y' - 2y = e^x")
solution2 = solve_linear_ode_step_by_step(eq2, y)
print(f"解为: {solution2}\n=========================================\n")




def laplace_determinant(M):
    """ 使用拉普拉斯展开式计算矩阵行列式 """
    # 当矩阵是1x1时
    if M.shape == (1, 1):
        return M[0, 0]
    # 当矩阵是2x2时
    if M.shape == (2, 2):
        return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    # 3x3或更大的矩阵递归计算
    det = 0
    for c in range(M.shape[#citation-1](citation-1)):
        submatrix = M.minor_submatrix(0, c)
        cofactor = ((-1) ** c) * M[0, c]
        sub_det = laplace_determinant(submatrix)
        det += cofactor * sub_det
        print(f"展开步骤: 去掉第0行、第{c}列得到子矩阵\n{submatrix}\n其行列式为 {sub_det}, 累加到行列式得到 det = {det}\n")
    return det
# 范例矩阵
B = sp.Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
det_B = laplace_determinant(B)
print("拉普拉斯展开式最终计算的行列式为: ")
print(det_B)



import sympy as sp
def gaussian_elimination(A, b):
    """ 使用高斯消元法解方程组 Ax = b """
    A = A.copy()
    b = b.copy()
    n = len(b)
    # 消元过程
    for i in range(n):
        # 寻找主元并进行行交换以确保稳定性
        max_row = max(range(i, n), key=lambda r: abs(A[r, i]))
        A[i, :], A[max_row, :] = A[max_row, :], A[i, :].copy()
        b[i], b[max_row] = b[max_row], b[i]
        print(f"第 {i+1} 步: 行交换，矩阵 A 和向量 b 调整为:\nA =\n{A},\nb =\n{b}\n")
        # 将i行以下的该列元素消为0
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
            print(f"消元操作: 消去第 {j+1} 行的第 {i+1} 列，调整为:\nA =\n{A},\nb =\n{b}\n")
    # 回代求解过程
    x = sp.zeros(n, 1)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:].dot(x[i + 1:])) / A[i, i]
        print(f"回代步骤: 计算 x[{i}]，结果为: x =\n{x}\n")
    return x
# 范例矩阵和向量
A = sp.Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = sp.Matrix([8, -11, -3])
x = gaussian_elimination(A, b)
print("高斯消元法最终解得: ")
sp.pprint(x)






import sympy as sp
# 定义符号变量
x = sp.symbols('x')
# ================================
# 1. 求导数
# ================================
# 定义函数
f = x**3 + 3*x**2 + 2*x + 1
# 求导
f_prime = sp.diff(f, x)
print("例题 1: 求函数 f(x) = x^3 + 3x^2 + 2x + 1 的导数")
print(f"导数: f'(x) = {f_prime}")
print("————————————————————————\n")
# ================================
# 2. 求定积分
# ================================
# 定义函数
f_integral = sp.exp(-x**2)
# 求定积分
integral_result = sp.integrate(f_integral, (x, -sp.oo, sp.oo))
print("例题 2: 计算函数 f(x) = e^(-x^2) 从 -∞ 到 ∞ 的定积分")
print(f"定积分: ∫ e^(-x²) dx from -∞ to ∞ = {integral_result}")
print("————————————————————————\n")
# ================================
# 3. 泰勒展开式
# ================================
# 定义函数
f_taylor = sp.sin(x)
# 泰勒展开
taylor_series = sp.series(f_taylor, x, 0, 6)
print("例题 3: 求 sin(x) 在 x=0 处的泰勒展开式 (展开到 x^5)")
print(f"泰勒展开式: {taylor_series}")
print("关键信息: 展开中心 = 0, 展开项数 = 到 x^5\n")
print("————————————————————————\n")
# ================================
# 4. 解微分方程
# ================================
# 定义函数y
y = sp.Function('y')
# 定义微分方程
deqn = sp.Eq(y(x).diff(x) + y(x), sp.sin(x))
# 求解微分方程
solution = sp.dsolve(deqn, y(x))
print("例题 4: 求解微分方程 y' + y = sin(x)")
print(f"通解: {solution}")
print("————————————————————————\n")
# ================================
# 5. 求极值
# ================================
# 定义函数
f2 = -x**4 + 4*x**3 + 2
# 求一阶导数
f2_prime = sp.diff(f2, x)
# 求临界点
critical_points = sp.solve(f2_prime, x)
print("例题 5: 找到函数 f(x) = -x^4 + 4x^3 + 2 的极值")
print(f"一阶导数 f'(x) = {f2_prime}")
print(f"临界点: {critical_points}")
# 二阶导数测试
f2_double_prime = sp.diff(f2_prime, x)
for point in critical_points:
    concavity = f2_double_prime.subs(x, point)
    if concavity < 0:
        max_min = "局部最大值"
    elif concavity > 0:
        max_min = "局部最小值"
    else:
        max_min = "拐点"
    print(f"x = {point}: {max_min}, 二阶导数 = {concavity}")
print("————————————————————————\n")




import sympy as sp

# 定义符号变量
x = sp.symbols('x')

# ================================
# 1. 向量运算
# ================================

v1 = sp.Matrix([2, -1, 4])
v2 = sp.Matrix([1, 3, 2])

print("1. 向量运算\n")

# 向量加法
v_add = v1 + v2
print("向量 v1 + v2 =")
sp.pprint(v_add)

# 向量点积
v_dot = v1.dot(v2)
print("向量 v1 ⋅ v2 =")
print(v_dot)

# 向量叉积
v_cross = v1.cross(v2)
print("向量 v1 x v2 =")
sp.pprint(v_cross)

print("——————————————————————————\n")

# ================================
# 2. 基本矩阵运算
# ================================

A = sp.Matrix([[4, 2], [1, 3]])
B = sp.Matrix([[0, -1], [2, 1]])

print("2. 基本矩阵运算\n")

# 矩阵加法
matrix_add = A + B
print("A + B =")
sp.pprint(matrix_add)

# 矩阵乘法
matrix_mult = A * B
print("A * B =")
sp.pprint(matrix_mult)

# 矩阵转置
A_transpose = A.transpose()
print("A 的转置 =")
sp.pprint(A_transpose)

print("——————————————————————————\n")

# ================================
# 3. 行列式与逆矩阵
# ================================

print("3. 行列式与逆矩阵\n")

# 行列式
det_A = A.det()
print("A 的行列式 det(A) =")
print(det_A)

# 逆矩阵
if det_A != 0:
    A_inv = A.inv()
    print("A 的逆矩阵 A^(-1) =")
    sp.pprint(A_inv)
else:
    print("A 不可逆。")

print("——————————————————————————\n")

# ================================
# 4. 特征值与特征向量
# ================================

print("4. 特征值与特征向量\n")

eigenvals = A.eigenvals()
print("特征值:")
sp.pprint(eigenvals)

eigenvects = A.eigenvects()
for val, mult, vects in eigenvects:
    print(f"\n特征值: {val} (重数: {mult})")
    for vect in vects:
        norm_vect = vect / vect.norm()
        print("特征向量 (归一化):")
        sp.pprint(norm_vect)

print("——————————————————————————\n")

# ================================
# 5. 矩阵分解
# ================================

print("5. 矩阵分解\n")

P, L, U = A.LUdecomposition()
print("LU 分解:")
print("P 矩阵 =")
sp.pprint(P)
print("L 矩阵 =")
sp.pprint(L)
print("U 矩阵 =")
sp.pprint(U)

Q, R = A.QRdecomposition()
print("\nQR 分解:")
print("Q 矩阵 =")
sp.pprint(Q)
print("R 矩阵 =")
sp.pprint(R)

print("——————————————————————————\n")

# ================================
# 6. 伴随矩阵与代数余子式
# ================================

print("6. 伴随矩阵与代数余子式\n")

# 伴随矩阵
adjugate_A = A.adjugate()
print("A 的伴随矩阵 =")
sp.pprint(adjugate_A)

# 代数余子式
minor_11 = A.minor(0, 0)
cofactor_11 = A.cofactor(0, 0)
print("\nA 的 (1,1) 元素的代数余子式: ")
print(f"余子式 = {minor_11}, 代数余子式 = {cofactor_11}")

print("——————————————————————————\n")

# ================================
# 7. 矩阵的秩与迹
# ================================

print("7. 矩阵的秩与迹\n")

# 矩阵的秩
rank_A = A.rank()
print("A 的秩 (rank) =")
print(rank_A)

# 矩阵的迹
trace_A = A.trace()
print("A 的迹 (trace) =")
print(trace_A)

print("——————————————————————————\n")

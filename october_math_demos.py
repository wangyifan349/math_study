#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sympy import (
    symbols, Function, diff, Eq, dsolve, pprint, init_printing,
    exp, sin, cos, series, limit, summation, oo, integrate,
    laplace_transform, fourier_transform, simplify, solve, factorial,
    Poly, Matrix, pi, erf, binomial, catalan, gamma, sqrt, log, Abs
)
from sympy.abc import x, n, t, s, k, m, y

init_printing(use_unicode=True)

# --------------------------------------------------------
# 1. 常微分方程
# --------------------------------------------------------
y_func = Function('y')

# ODE: 齐次方程 y'' - 5y' + 6y = 0
ode1 = Eq(diff(y_func(x), x, 2) - 5*diff(y_func(x), x) + 6*y_func(x), 0)
sol1 = dsolve(ode1, y_func(x))
pprint("ODE 1: y'' - 5y' + 6y = 0")
pprint(sol1)

# ODE: 初值问题 y'' + 4y = 0, y(0)=1, y'(0)=0
ode2 = Eq(diff(y_func(x), x, 2) + 4*y_func(x), 0)
sol2 = dsolve(ode2, y_func(x), ics={y_func(0): 1, diff(y_func(x), x).subs(x, 0): 0})
pprint("ODE 2: y'' + 4y = 0, y(0)=1, y'(0)=0")
pprint(sol2)

# ODE: 非齐次方程 y'' - y = exp(x)
ode3 = Eq(diff(y_func(x), x, 2) - y_func(x), exp(x))
sol3 = dsolve(ode3, y_func(x))
pprint("ODE 3: y'' - y = exp(x)")
pprint(sol3)

# Euler-Cauchy 方程: x²*y'' - 3x*y' + 4y = 0
ode4 = Eq(x**2*diff(y_func(x), x, 2) - 3*x*diff(y_func(x), x) + 4*y_func(x), 0)
sol4 = dsolve(ode4, y_func(x))
pprint("Euler-Cauchy: x²*y'' - 3x*y' + 4y = 0")
pprint(sol4)

# --------------------------------------------------------
# 2. 偏微分方程（PDE）示例
# 1阶线性 PDE: u_x + u_y = 0
u = Function('u')
# 形式解：u(x,y)= f(x-y) 或 f(x+y)等
pprint("PDE: u_x + u_y = 0, 通解形式 u(x,y)= f(x-y)")
pprint("示例解: u(x,y)= exp(-(x-y))")  # 形式解示例

# 2阶PDE: 热方程 u_t = k*u_xx; 令 k=1
u = Function('u')
pde_heat = Eq(diff(u(x, t), t), diff(u(x, t), x, 2))
pprint("Heat 方程: u_t = u_xx (形式解)")
pprint("解法通常采用分离变量等方法，此处略。")

# --------------------------------------------------------
# 3. 积分、级数和特殊函数
# --------------------------------------------------------
# 3.1 不定积分 ∫ sin(x)*exp(x) dx
pprint("Indefinite Integral: ∫ sin(x)*exp(x) dx")
pprint(integrate(sin(x)*exp(x), x))

# 3.2 定积分 ∫₀^π sin(x) dx
pprint("Definite Integral: ∫₀^π sin(x) dx")
pprint(simplify(integrate(sin(x), (x, 0, pi))))

# 3.3 积分 by parts 示例: ∫ x*cos(x) dx
pprint("Integration by parts: ∫ x*cos(x) dx")
pprint(integrate(x*cos(x), x))

# 3.4 Taylor 展开 exp(x) 展开到 x^5
pprint("Taylor: exp(x) 展开到 x^5")
pprint(series(exp(x), x, 0, 6))

# 3.5 Taylor 展开 sin(x) 展开到 x^7
pprint("Taylor: sin(x) 展开到 x^7")
pprint(series(sin(x), x, 0, 8))

# 3.6 极限：lim(x->0) sin(x)/x
pprint("Limit: sin(x)/x as x->0")
pprint(limit(sin(x)/x, x, 0))

# 3.7 级数求和：∑ₙ₌₁∞ 1/n²
pprint("Series: ∑ₙ₌₁∞ 1/n²")
pprint(summation(1/n**2, (n, 1, oo)))

# 3.8 部分和证明 exp(x) = ∑ₙ₌₀∞ xⁿ/n! (前6项)
pprint("Partial Sum for exp(x) (n=0..5)")
pprint(summation(x**n/factorial(n), (n, 0, 5)))

# 3.9 特殊函数：Gamma, Erf, Catalan 数
pprint("Gamma(5):")
pprint(gamma(5))
pprint("erf(x):")
pprint(erf(x))
pprint("Catalan Number C(4):")
pprint(catalan(4))

# --------------------------------------------------------
# 4. 多项式操作
# --------------------------------------------------------
# 多项式: P(x)= x³ - 2x² + 4x - 8
P = Poly(x**3 - 2*x**2 + 4*x - 8, x)
pprint("Polynomial Factorization:")
pprint(P.factor_list())
pprint("Polynomial Roots:")
pprint(P.nroots())

# --------------------------------------------------------
# 5. 矩阵和线性代数
# --------------------------------------------------------
# 矩阵运算
A = Matrix([[2, 3, 1],
            [4, 1, -1],
            [3, -2, 5]])
pprint("Matrix A:")
pprint(A)
pprint("Determinant of A:")
pprint(A.det())
if A.det() != 0:
    pprint("Inverse of A:")
    pprint(A.inv())

# 5.1 解线性方程组: 3x + 2y = 12, x - y = 1
x_sym, y_sym = symbols('x y')
sol_lin = solve((Eq(3*x_sym + 2*y_sym, 12), Eq(x_sym - y_sym, 1)), (x_sym, y_sym))
pprint("Linear System Solution:")
pprint(sol_lin)

# 5.2 矩阵 A2 的特征值与特征向量
A2 = Matrix([[1, 2], [3, 4]])
pprint("Matrix A2:")
pprint(A2)
pprint("Eigenvalues and eigenvectors of A2:")
pprint(A2.eigenvects())

# --------------------------------------------------------
# 6. 组合数学与数论
# --------------------------------------------------------
# 二项式系数
pprint("Binomial Coefficient: C(5,2)")
pprint(binomial(5,2))
# 计算阶乘
pprint("Factorial: 7!")
pprint(factorial(7))
# 简单同余示例：判断 17 是否为质数
def is_prime(p):
    if p <= 1:
        return False
    for i in range(2, int(sqrt(p)) + 1):
        if p % i == 0:
            return False
    return True

prime_check = is_prime(17)
pprint("Is 17 prime?")
pprint(prime_check)

# --------------------------------------------------------
# 7. 概率与统计（符号积分形式求期望）
# --------------------------------------------------------
# 简单概率密度函数：f(x) = 2*x, 0<= x <=1, 求期望 E[X]
f_pdf = 2*x
E_X = integrate(x*f_pdf, (x, 0, 1))
pprint("Expected value E[X] for f(x)=2x on [0,1]:")
pprint(E_X)

# --------------------------------------------------------
# 8. 其他数学表达
# --------------------------------------------------------
# 计算对数、绝对值、平方根
expr = log(Abs(x)) + sqrt(x)
pprint("Expression: log(|x|) + sqrt(x)")
pprint(expr)
pprint("Simplified expression for x>0 with x=9:")
pprint(expr.subs(x, 9))
pprint("Numerical value:")
pprint(expr.subs(x, 9).evalf())

pprint("所有计算结束.")




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sympy import symbols, Function, Eq, diff, dsolve, pprint, init_printing, exp, sin, cos, series, limit, summation, oo, integrate, laplace_transform, fourier_transform, simplify, solve, factorial, Poly, Matrix, pi, erf, binomial, catalan, gamma, sqrt, log, Abs, residue, apart, polar_lift, factor, collect, zeta, polylog, expand, Heaviside, DiracDelta
from sympy.abc import x, n, t, s, k, y, z
init_printing(use_unicode=True)
# 1. 常微分方程
y_func = Function('y')
ode1 = Eq(diff(y_func(x), x, 2) - 5*diff(y_func(x), x) + 6*y_func(x), 0)
sol1 = dsolve(ode1, y_func(x))
pprint("ODE 1: y'' - 5y' + 6y = 0")
pprint(sol1)
ode2 = Eq(diff(y_func(x), x, 2) + 4*y_func(x), 0)
sol2 = dsolve(ode2, y_func(x), ics={y_func(0): 1, diff(y_func(x), x).subs(x,0): 0})
pprint("ODE 2: y'' + 4y = 0, y(0)=1, y'(0)=0")
pprint(sol2)
ode3 = Eq(diff(y_func(x), x, 2) - y_func(x), exp(x))
sol3 = dsolve(ode3, y_func(x))
pprint("ODE 3: y'' - y = exp(x)")
pprint(sol3)
ode4 = Eq(x**2*diff(y_func(x), x, 2) - 3*x*diff(y_func(x), x) + 4*y_func(x), 0)
sol4 = dsolve(ode4, y_func(x))
pprint("Euler-Cauchy: x²*y'' - 3x*y' + 4y = 0")
pprint(sol4)
# 2. 偏微分方程（PDE）示例
u = Function('u')
pprint("PDE: u_x + u_y = 0, 通解形式 u(x,y)= f(x-y)")
pprint("示例解: u(x,y)= exp(-(x-y))")
pprint(exp(-(x-y)))
# 3. 积分、级数与特殊函数
pprint("Indefinite Integral: ∫ sin(x)*exp(x) dx")
pprint(integrate(sin(x)*exp(x), x))
pprint("Definite Integral: ∫₀^π sin(x) dx")
pprint(simplify(integrate(sin(x), (x, 0, pi))))
pprint("Integration by parts: ∫ x*cos(x) dx")
pprint(integrate(x*cos(x), x))
pprint("Taylor: exp(x) 展开到 x^5")
pprint(series(exp(x), x, 0, 6))
pprint("Taylor: sin(x) 展开到 x^7")
pprint(series(sin(x), x, 0, 8))
pprint("Limit: sin(x)/x as x->0")
pprint(limit(sin(x)/x, x, 0))
pprint("Series: ∑ₙ₌₁∞ 1/n²")
pprint(summation(1/n**2, (n, 1, oo)))
pprint("Partial Sum for exp(x) (n=0..5)")
pprint(summation(x**n/factorial(n), (n, 0, 5)))
pprint("Gamma(5):")
pprint(gamma(5))
pprint("erf(x):")
pprint(erf(x))
pprint("Catalan Number C(4):")
pprint(catalan(4))
# 4. 多项式操作
P = Poly(x**3 - 2*x**2 + 4*x - 8, x)
pprint("Polynomial Factorization:")
pprint(P.factor_list())
pprint("Polynomial Roots:")
pprint(P.nroots())
# 5. 矩阵与线性代数
A = Matrix([[2,3,1],[4,1,-1],[3,-2,5]])
pprint("Matrix A:")
pprint(A)
pprint("Determinant of A:")
pprint(A.det())
if A.det() != 0:
    pprint("Inverse of A:")
    pprint(A.inv())
x_sym, y_sym = symbols('x y')
sol_lin = solve((Eq(3*x_sym+2*y_sym,12), Eq(x_sym-y_sym,1)), (x_sym, y_sym))
pprint("Linear System Solution:")
pprint(sol_lin)
A2 = Matrix([[1,2],[3,4]])
pprint("Matrix A2:")
pprint(A2)
pprint("Eigenvalues and eigenvectors of A2:")
pprint(A2.eigenvects())
# 6. 复变函数与留数
z_var = symbols('z')
f_z = exp(z_var)/(z_var**2*(z_var-1))
pprint("Residue of f(z)=exp(z)/(z²(z-1)) at z=0:")
pprint(residue(f_z, z_var, 0))
pprint("Residue at z=1:")
pprint(residue(f_z, z_var, 1))
pprint("Partial fraction of f(z):")
pprint(apart(f_z, z_var))
# 7. 多变量微积分
x1, x2 = symbols('x1 x2')
f_multi = x1**2*sin(x2) + exp(x1*x2)
pprint("∂f/∂x1:")
pprint(diff(f_multi, x1))
pprint("Gradient of f_multi:")
pprint([diff(f_multi, var) for var in (x1,x2)])
pprint("Hessian of f_multi:")
pprint(Matrix([[diff(f_multi, var1, var2) for var2 in (x1,x2)] for var1 in (x1,x2)]))
# 8. 高阶导数和 Laurent 展开
pprint("5th derivative of 1/(1-x):")
pprint(diff(1/(1-x), x, 5))
pprint("Laurent expansion of exp(1/x) around x=0 up to order 7:")
pprint(series(exp(1/x), x, 0, 7))
# 9. 极坐标转换
from sympy import atan2
z_expr = 3+4*1j
pprint("Polar representation (using polar_lift):")
pprint(polar_lift(z_expr))
pprint("Modulus and argument:")
pprint(Abs(z_expr))
pprint(simplify(atan2(4,3)))
# 10. Riemann ζ 与 Polylog
pprint("Riemann ζ(2):")
pprint(zeta(2))
pprint("Polylog: Li₂(1/2):")
pprint(polylog(2, 1/2))
# 11. 分段函数和 DiracDelta 积分
f_piecewise = Heaviside(x-1) - Heaviside(x-2)
pprint("Integral of piecewise function ∫₀³ [H(x-1)-H(x-2)] dx:")
pprint(integrate(f_piecewise, (x, 0, 3)))
pprint("Integral of DiracDelta(x-2)*sin(x) from 0 to 5:")
pprint(integrate(DiracDelta(x-2)*sin(x), (x, 0, 5)))
# 12. 非线性方程求解
pprint("Solve nonlinear equation: x³ - x - 2 = 0")
pprint(solve(x**3 - x - 2, x))
# 13. 多项式进阶操作
P_poly = Poly(x**4 - 3*x**3 + x - 5, x)
pprint("Factorization of P(x):")
pprint(factor(P_poly.as_expr()))
pprint("Collect like terms for P(x):")
pprint(collect(P_poly.as_expr(), x))
# 14. 符号矩阵的逆与特征分解
a,b,c,d = symbols('a b c d')
M = Matrix([[a, b],[c, d]])
pprint("Symbolic Matrix M:")
pprint(M)
pprint("Inverse of M:")
pprint(M.inv())
pprint("Eigen decomposition of M:")
pprint(M.eigenvects())
# 15. 组合数学和数论
pprint("Binomial Coefficient C(5,2):")
pprint(binomial(5,2))
pprint("7! =")
pprint(factorial(7))
pprint("Expansion of (x+y)^5:")
pprint(expand((x+y)**5))
pprint("Catalan number for n=10:")
pprint(catalan(10))
pprint("All computations complete.")




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sympy import (
    symbols, Function, diff, Eq, dsolve, pprint, init_printing,
    exp, sin, cos, series, limit, summation, oo
)
from sympy.abc import x, n

# Initialize pretty printing (uses Unicode for displaying formulas beautifully)
init_printing(use_unicode=True)
# --------------------------------------------------
# 1. Example of a Constant Coefficient Differential Equation
#
# We want to solve the second-order homogeneous differential equation:
#   y'' - 5y' + 6y = 0
# The corresponding characteristic equation is r² - 5r + 6 = 0,
# which factors to (r - 2)(r - 3) = 0. Thus, the roots are r = 2 and r = 3.
# The general solution is a combination of exponential functions:
#   y = C1 * exp(2x) + C2 * exp(3x)
# --------------------------------------------------
y_func = Function('y')
ode_const_coeff = Eq(diff(y_func(x), x, 2) - 5*diff(y_func(x), x) + 6*y_func(x), 0)
pprint(ode_const_coeff)
sol_const_coeff = dsolve(ode_const_coeff, y_func(x))
print("\nGeneral solution of the constant coefficient homogeneous differential equation:")
pprint(sol_const_coeff)

# --------------------------------------------------
# 2. Constant Coefficient Differential Equation with Initial Conditions
#
# Solve the initial value problem:
#   y'' + 4y = 0, with initial conditions y(0) = 1 and y'(0) = 0.
# The characteristic equation is r² + 4 = 0, yielding roots r = ±2i,
# indicating a solution involving sine and cosine:
#   y = C1 * cos(2x) + C2 * sin(2x)
# Apply the initial conditions to find the specific constants.
# --------------------------------------------------
ode_ivp = Eq(diff(y_func(x), x, 2) + 4*y_func(x), 0)
pprint(ode_ivp)
sol_ivp = dsolve(ode_ivp, y_func(x), ics={y_func(0): 1, diff(y_func(x), x).subs(x, 0): 0})
print("\nUnique solution of the constant coefficient differential equation with initial conditions:")
pprint(sol_ivp)

# --------------------------------------------------
# 3. Other Mathematical Operations
#    3.1 Taylor Series: Expand exp(x) at x=0 up to x^5
#         The Taylor series of exp(x) is a sum of terms with increasing powers of x
#    3.2 Compute a Limit: lim(x->0) sin(x)/x
#         A fundamental limit in calculus, should equal 1
#    3.3 Series Sum: Σ 1/n² (n from 1 to ∞)
#         This is known as the Basel problem, and the sum converges to π²/6
# --------------------------------------------------
print("\nTaylor series of exp(x) up to x^5:")
pprint(series(exp(x), x, 0, 6))

print("\nlim(x->0) sin(x)/x =")
pprint(limit(sin(x)/x, x, 0))

print("\nSeries ∑ₙ₌₁∞ 1/n² =")
pprint(summation(1/n**2, (n, 1, oo)))

print("\nEnd of all examples.")

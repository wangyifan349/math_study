import numpy as np

# Calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)  # Calculate the dot product
    norm_vec1 = np.linalg.norm(vec1)  # Calculate the norm of the first vector
    norm_vec2 = np.linalg.norm(vec2)  # Calculate the norm of the second vector
    similarity = dot_product / (norm_vec1 * norm_vec2)  # Compute cosine similarity
    return similarity

# Example usage
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
similarity = cosine_similarity(vec1, vec2)
print("Cosine similarity:", similarity)
# --------------------------------------------
import numpy as np

# Calculate Euclidean distance (L2 distance) between two vectors
def euclidean_distance(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)  # Calculate the L2 distance
    return distance

# Example usage
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
distance = euclidean_distance(vec1, vec2)
print("L2 (Euclidean) distance:", distance)
# --------------------------------------------
import numpy as np

# Calculate determinant of a matrix
A = np.array([[1, 2],
              [3, 4]])

det_A = np.linalg.det(A)
print("Determinant of matrix:", det_A)
# --------------------------------------------
import numpy as np

# Calculate eigenvalues and eigenvectors of a matrix
A = np.array([[1, 2],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
# --------------------------------------------
import numpy as np

# Perform singular value decomposition (SVD) on a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, S, Vt = np.linalg.svd(A)
print("U matrix:")
print(U)
print("Singular values:")
print(S)
print("Vt matrix:")
print(Vt)
# --------------------------------------------
import numpy as np

# Perform Fast Fourier Transform (FFT) on a signal
signal = np.array([1, 2, 0, -1, -1.5, -2, 1, 1.5])

fft_result = np.fft.fft(signal)
print("FFT result:")
print(fft_result)
# --------------------------------------------
import numpy as np
from scipy import stats

# Perform linear regression using example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("Slope:", slope)
print("Intercept:", intercept)
print("R value:", r_value)
# --------------------------------------------
import numpy as np
from sklearn.decomposition import PCA

# Perform Principal Component Analysis (PCA) on example data
data = np.array([[2.5, 2.4],
                 [0.5, 0.7],
                 [2.2, 2.9],
                 [1.9, 2.2],
                 [3.1, 3.0],
                 [2.3, 2.7],
                 [2, 1.6],
                 [1, 1.1],
                 [1.5, 1.6],
                 [1.1, 0.9]])

pca = PCA(n_components=2)
pca.fit(data)

print("Principal components:")
print(pca.components_)
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
# --------------------------------------------
import numpy as np

# Estimate numerical derivative of a function at given point
def f(x):
    return np.sin(x)

x = np.pi / 4
h = 1e-5  # A small perturbation
derivative = (f(x + h) - f(x - h)) / (2 * h)
print("Approximate derivative of sin(x) at x = pi/4:", derivative)
# --------------------------------------------
import numpy as np
from scipy import integrate

# Estimate numerical integral of a function over an interval
def f(x):
    return np.exp(-x**2)

integral, error = integrate.quad(f, 0, 1)
print("Integral of exp(-x^2) from 0 to 1:", integral)
# --------------------------------------------
import numpy as np
from scipy.integrate import solve_ivp

# Solve an ordinary differential equation (ODE) dy/dt = -2y
def dydt(t, y):
    return -2 * y

t_span = (0, 5)  # Time interval
y0 = [1]  # Initial condition

solution = solve_ivp(dydt, t_span, y0, method='RK45', t_eval=np.linspace(0, 5, 100))
print("Time points:", solution.t)
print("Values of y:", solution.y[0])
# --------------------------------------------
import numpy as np
from scipy import integrate

# Compute double integral of a function over a region
def f(x, y):
    return np.exp(-x**2 - y**2)

def gfun(x):
    return 0  # Lower limit for y

def hfun(x):
    return 1  # Upper limit for y

integral, error = integrate.dblquad(f, 0, 1, gfun, hfun)
print("Double integral of exp(-x^2 - y^2) over [0,1]x[0,1]:", integral)

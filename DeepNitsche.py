# Import libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from collections import OrderedDict
from scipy.special import roots_legendre
from scipy.stats import uniform
import warnings
import time
import os
import csv
from datetime import datetime
from utils import *

warnings.filterwarnings('ignore', category=UserWarning)

# set seed function
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) # for CPU
    if torch.cuda.is_available(): # for GPU
        torch.cuda.manual_seed_all(seed)

# the unit normal vector at points on the interface
def normal_vector(x, y):
    dist = np.sqrt(x ** 2 + y ** 2)
    normal_x = x / dist
    normal_y = y / dist
    normal = np.column_stack((normal_x, normal_y))
    return normal

def sign_x(x, y):
    z = np.ones_like(x)
    dist_squared = x ** 2 + y ** 2
    z[dist_squared < r0 ** 2] = -1.0
    return z


set_seed(42)
# beta define
# (beta_plus, beta_minus)
beta_plus = 10.0
beta_minus = 1.0
alpha = 1.0
r0 = 0.5


def exact_u(x, y, z):
    r_squared = x ** 2 + y ** 2
    u_plus = r_squared / (2.0 * beta_plus)
    u_minus = r_squared / (2.0 * beta_minus) - r0 ** alpha + (1 / beta_plus - 1 / beta_minus) * r0 ** 2 / 2
    eu = u_plus * (1.0 + z) / 2.0 + u_minus * (1.0 - z) / 2.0
    return eu

# the gradient of the exact_u
def exact_du(x, y, z):
    ux_plus, uy_plus = x, y
    ux_minus, uy_minus = x, y
    dux = ux_plus * (1.0 + z) / 2.0 + ux_minus * (1.0 - z) / 2.0
    duy = uy_plus * (1.0 + z) / 2.0 + uy_minus * (1.0 - z) / 2.0
    return dux, duy

# V define. [u] =u_plus - u_minus  = v
def V(x_interface, y_interface):
    '''
    [u] = u_minus - u_plus
    '''
    r_squared = x_interface ** 2 + y_interface ** 2
    
    # 在界面上的精确解值
    u_plus = r_squared / (2.0 * beta_plus)
    u_minus = r_squared / (2.0 * beta_minus) - r0 ** alpha + (1 / beta_plus - 1 / beta_minus) * r0 ** 2 / 2
    
    # 跳跃值
    jump = u_minus - u_plus
    return jump

def W(x_interface, y_interface):
    '''
    通用W函数：基于精确解计算法向通量跳跃值
    [β∇u·n] = β_minus*(∂u_minus/∂n) - β_plus*(∂u_plus/∂n)
    '''
    # 对于精确解 u = r²/(2β)，在圆形界面上：
    # ∂u/∂n = ∇u·n = (x/β)(x/r0) + (y/β)(y/r0) = r0/β
    
    dudn_plus = r0 / beta_plus
    dudn_minus = r0 / beta_minus
    
    # 法向通量跳跃
    flux_jump = beta_minus * dudn_minus - beta_plus * dudn_plus
    
    return np.full_like(x_interface, flux_jump)


def rF(x, y, z):
    # # for u = r²/2， Δu = 2，so -Δu = -2
    f_plus = -2.0 + x * 0 + y * 0
    f_minus = -2.0 + x * 0 + y * 0
    rf = f_plus * (1.0 + z) / 2.0 + f_minus * (1.0 - z) / 2.0
    return rf

# the computation of boundary normal vector
def compute_boundary_normal(X_bd, x_range=(-1, 1), y_range=(-1, 1), tol=1e-6):
    # no corner point
    normals = torch.zeros_like(X_bd[:, :2])
    x = X_bd[:, 0]
    y = X_bd[:, 1]
    
    # Points on the four edges (no need to exclude corner points)
    bottom_edge = torch.abs(y - y_range[0]) < tol  # Bottom boundary y = -1
    top_edge = torch.abs(y - y_range[1]) < tol     # Top boundary y = 1  
    left_edge = torch.abs(x - x_range[0]) < tol    # Left boundary x = -1
    right_edge = torch.abs(x - x_range[1]) < tol   # Right boundary x = 1
    
    # Set normal vectors for each edge (pointing outward)
    normals[bottom_edge] = torch.tensor([0.0, -1.0], dtype=X_bd.dtype, device=X_bd.device)
    normals[top_edge] = torch.tensor([0.0, 1.0], dtype=X_bd.dtype, device=X_bd.device)
    normals[left_edge] = torch.tensor([-1.0, 0.0], dtype=X_bd.dtype, device=X_bd.device)
    normals[right_edge] = torch.tensor([1.0, 0.0], dtype=X_bd.dtype, device=X_bd.device)

    return normals


device = torch.device("cpu")

# Define model
class Plain(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.ln1 = nn.Linear(in_dim, h_dim).double()
        self.act1 = nn.Tanh()
        self.ln2 = nn.Linear(h_dim, h_dim).double()
        self.act2 = nn.Tanh()
        self.ln3 = nn.Linear(h_dim, out_dim, bias=True).double()

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.ln3(out)
        return out
    

# Simple method
# Chebyshev_first_kind
def chebyshev_first_kind(dim, n):
    X = []
    x = []
    X = (np.mgrid[[slice(None, n), ] * dim])
    XX = np.cos(np.pi * (X + 0.5) / n)
    for i in range(len(X)):
        x.append(np.array(XX[i].tolist()).reshape(n ** dim, 1))
    return np.hstack(np.array(x))

# Gauss_Quadrature
def gauss_nodes_multi(dim, n):
    nodes = np.zeros((n ** dim, dim))
    weights = np.zeros(n ** dim)

    if dim == 1:
        X, W = roots_legendre(n)
        nodes[:, 0] = X
        weights[:] = W
    elif dim == 2:
        X, W = roots_legendre(n)
        for i in range(n):
            for j in range(n):
                nodes[i * n + j, 0] = X[i]
                nodes[i * n + j, 1] = X[j]
                weights[i * n + j] = W[i] * W[j]
    else:
        raise ValueError("only support 1D and 2D Gauss nodes")

    return nodes, weights

def uniform_nodes(dim, n):
    X = np.linspace(-1, 1, n)
    nodes = np.meshgrid(*[X] * dim)
    return np.vstack([node.flatten() for node in nodes]).T

# Collect training points
# Training points
# X_inner: points inside the domain, totally (N_inner)**2 points
def gauss_nodes_1d(n):
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(n)
    return nodes, weights


# Loss function
def loss(model, X_inner, weights, Rf_inner, X_bd, U_bd, X_interface, Normal_interface, Jump, Jump_normal, N_inner):
    # boundary prediction
    bd_pred = model(X_bd)
    boundary_rhs = torch.mean(U_bd * bd_pred)

    # Nitsche penalty parameter
    eta = 2000.0

    # boundart gradient
    X_bd.requires_grad = True
    bd_pred_grad = model(X_bd)
    grad_u_bd = torch.autograd.grad(
        bd_pred_grad, X_bd,
        grad_outputs=torch.ones_like(bd_pred_grad),
        retain_graph=True,
        create_graph=True
    )[0]
    # boundary normal vector
    normals_bd = compute_boundary_normal(X_bd)

    # normal difference
    normal_diff = beta_plus * (
        grad_u_bd[:, 0:1] * normals_bd[:, 0:1] +
        grad_u_bd[:, 1:2] * normals_bd[:, 1:2]
    )
    # boundary consistency term
    boundary_length = 8 # the length of the boundary
    boundary_consistency_term = boundary_length * torch.mean(normal_diff * U_bd)

    # inner prediction
    inner_pred = model(X_inner)
    dudX = torch.autograd.grad(
        inner_pred, X_inner,
        grad_outputs=torch.ones_like(inner_pred),
        retain_graph=True,
        create_graph=True
    )[0]

    # process the interface
    normal_x = Normal_interface[:, 0:1]
    normal_y = Normal_interface[:, 1:2]
    # ensure the gradient of the interface is true
    interface_outer = torch.cat([X_interface[:, 0:2], torch.ones(X_interface.shape[0], 1, device=X_interface.device, requires_grad=True)], dim=1)
    interface_inner = torch.cat([X_interface[:, 0:2], -torch.ones(X_interface.shape[0], 1, device=X_interface.device, requires_grad=True)], dim=1)

    u_interface_outer = model(interface_outer)
    u_interface_inner = model(interface_inner)

    # gradients at the computing interface
    ux_interface_outer = torch.autograd.grad(
        u_interface_outer, interface_outer,
        grad_outputs=torch.ones_like(u_interface_outer),
        retain_graph=True,
        create_graph=True
        )[0] 

    ux_interface_inner = torch.autograd.grad(
        u_interface_inner, interface_inner,
        grad_outputs=torch.ones_like(u_interface_inner),
        retain_graph=True,
        create_graph=True
        )[0] 

    # normal vector at the interface
    Normal_outer = normal_x * ux_interface_outer[:, 0:1] + normal_y * ux_interface_outer[:, 1:2]
    Normal_inner = normal_x * ux_interface_inner[:, 0:1] + normal_y * ux_interface_inner[:, 1:2] 

    # beta value
    z_values = X_inner[:, 2]
    beta_values = torch.where(z_values < 0, beta_minus, beta_plus) 

    
    # β|∇u|² grad_coupling_term
    grad_u_square = beta_values * (dudX[:, 0] * dudX[:, 0] + dudX[:, 1] * dudX[:, 1])
    weights_normalized = weights * 4.0 / torch.sum(weights) 
    grad_coupling_term =  torch.dot(grad_u_square, weights_normalized) 

    normal_jump_pred = beta_minus * Normal_inner - beta_plus * Normal_outer

    jump_pred = u_interface_inner - u_interface_outer  # [u] = u⁻ - u⁺
    
    N_interface_points = X_interface.shape[0]
    interface_length = 2 * np.pi * r0
    ds = interface_length / N_interface_points
    gamma = 1.0 
    line_integral = gamma / alpha * torch.sum(jump_pred ** 2) * ds # interface_jump
    # line_integral = gamma / alpha * torch.sum(jump_pred ** 2) * ds # interface_jump

    

    fv = torch.dot(Rf_inner.squeeze() * inner_pred.squeeze(), weights_normalized)
    source_term = -fv 
    

    # Boundary flux
    boundary_flux_term = - 2.0 * boundary_length * torch.mean(normal_diff * bd_pred_grad) 
    

    # Interface stability term
    boundary_stable = eta * boundary_length * torch.mean(bd_pred * bd_pred) # torch.Size([])


    # bilnear term
    bilinear_term = 0.5 * (grad_coupling_term + line_integral + boundary_flux_term + boundary_stable)
    # bilinear_term = 0.5 * (grad_coupling_term + boundary_flux_term + boundary_stable) 
    
    
    flux_condition_residual = normal_jump_pred
    loss_normal_jump = torch.mean(flux_condition_residual ** 2)

    # Boundary penalty term
    boundary_penalty_term = - eta * boundary_length * boundary_rhs

    # Total loss
    L1 = bilinear_term + source_term + boundary_consistency_term + boundary_penalty_term
    # L2 = loss_normal_jump

    # total_loss = L1 + eta * L2
    # total_loss = L1 + eta * loss_normal_jump
    total_loss = L1


    return total_loss

# Number of grid points per quadrant
quarant_number_one_side = 64 # 16, 24, 32, 48, 64

N_inner_quadrant = 1 
# number of grid points
N_inner = quarant_number_one_side * N_inner_quadrant

# Define boundaries of each quadrant
quadrant_boundaries = np.zeros([N_inner_quadrant, 4]) # (1, 4)
for i in range(N_inner_quadrant):
    quadrant_boundaries[i] = np.array([-1., -1., 1., 1.], dtype=float)
    # [[-1., -1., 1., 1.]]
    # len(quadrant_boundaries): 1

X_inner_quadrants = [] # (256, 2)
weights_inner_quadrants = []

for quadrant_boundary in quadrant_boundaries:
    x_min, y_min, x_max, y_max = quadrant_boundary
    
    x_nodes, x_weights = gauss_nodes_1d(quarant_number_one_side) # (24,)
    y_nodes, y_weights = gauss_nodes_1d(quarant_number_one_side)

    x_nodes = 0.5 * (x_nodes + 1) * (x_max - x_min) + x_min
    y_nodes = 0.5 * (y_nodes + 1) * (y_max - y_min) + y_min
    x_weights = x_weights * (x_max - x_min) / 2
    y_weights = y_weights * (y_max - y_min) / 2

    xx, yy = np.meshgrid(x_nodes, y_nodes)

    X_inner_quadrants.append(np.column_stack((xx.flatten(), yy.flatten())))

    weights_2d = x_weights.reshape(-1, 1) * y_weights
    weights_flat = weights_2d.flatten() 
    weights_inner_quadrants.append(weights_flat)

# Combine points from all quadrants
X_inner = np.vstack(X_inner_quadrants)
weights_inner = np.hstack(weights_inner_quadrants)

# Calculate the sum of inner weights
sum_weights_inner = np.sum(weights_inner)

# Calculate Rf_inner using rf_u function
x = X_inner[:, 0:1] 
y = X_inner[:, 1:2] 
z = sign_x(x, y) 
X_inner = np.hstack((X_inner, z)) 
Rf_inner = rF(x, y, z) 

# X_bd: points on the boundary, totally 4*N_inner points
# uniform_point = uniform_nodes(1, N_inner)
gauss_point, _ = gauss_nodes_1d(N_inner)
gauss_point = gauss_point.reshape(-1, 1)
dumy_one = np.ones((N_inner, 1))

xx1 = np.hstack((gauss_point, -1.0 * dumy_one, dumy_one))
xx2 = np.hstack((-1.0 * dumy_one, gauss_point, dumy_one)) 
xx3 = np.hstack((dumy_one, gauss_point, dumy_one)) 
xx4 = np.hstack((gauss_point, dumy_one, dumy_one)) 
X_bd = np.vstack([xx1, xx2, xx3, xx4]) 
U_bd = exact_u(X_bd[:, 0], X_bd[:, 1], X_bd[:, 2]).reshape(-1, 1)

# Number of points on the interior interface
N_interface = 8 * N_inner  # Adjust this if necessary

# Generate evenly spaced angles between 0 and 2π
theta = np.linspace(0, 2 * np.pi, N_interface, endpoint=False)
# 

# Calculate (x, y) coordinates of the points on the circle of radius 0.5
x_interface = r0 * np.cos(theta) 
y_interface = r0 * np.sin(theta) 
# Stack the coordinates to form the interface points
X_interface = np.vstack((x_interface, y_interface)).T 

# normal vector
Normal_interface = normal_vector(x_interface, y_interface) 

## Jump: function jump on the interior interface, totally 4*N_inner points
Jump = V(x_interface, y_interface) # (64,)

# Jump_normal: normal jump on the interior interface
Jump_normal = W(x_interface, y_interface) # (64,)

# Switch variables to torch
X_bd_torch = torch.tensor(X_bd, requires_grad=True, device=device, dtype=torch.float64)
U_bd_torch = torch.tensor(U_bd, device=device, dtype=torch.float64)
X_inner_torch = torch.tensor(X_inner, requires_grad=True, device=device, dtype=torch.float64)
weights_inner_torch = torch.tensor(weights_inner).double().to(device)
Rf_inner_torch = torch.tensor(Rf_inner, device=device, dtype=torch.float64)
X_interface_torch = torch.tensor(X_interface, requires_grad=True, device=device, dtype=torch.float64)
Normal_interface_torch = torch.tensor(Normal_interface, device=device, dtype=torch.float64)
Jump_torch = torch.tensor(Jump, device=device, dtype=torch.float64)
Jump_normal_torch = torch.tensor(Jump_normal, device=device, dtype=torch.float64)

# Training points plot
plt.figure(figsize=(5, 5))

plt.scatter(X_inner[:, 0], X_inner[:, 1],
            c="b", s=1, marker=".")
plt.scatter(X_bd[:, 0], X_bd[:, 1],
            c="r", s=5, marker=".")
plt.scatter(X_interface[:, 0], X_interface[:, 1],
            c="k", s=5, marker="o")
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.title('training points')
plt.savefig('results_DeepNitsche/Training points.png', dpi=300)
plt.show()
plt.close()

# Generate validation points (Improved verification point generation strategy)
N_valid_inner = int(0.15 * X_inner.shape[0])  # Reduce random internal points
N_valid_interface = int(0.05 * X_inner.shape[0])  # Fix: based on X_inner instead of X_interface

# 1. Random internal validation points
valid_indices = np.random.choice(X_inner.shape[0], N_valid_inner, replace=False)
X_valid_inner_random = X_inner[valid_indices, :]
weights_valid_inner_random = weights_inner[valid_indices]
Rf_valid_inner_random = Rf_inner[valid_indices]

# 2. Structured validation points near the interface
np.random.seed(123)  # Ensure reproducibility
theta_valid = np.linspace(0, 2 * np.pi, N_valid_interface, endpoint=False)

# Inner circle validation points (r = 0.3 to 0.48)
r_valid_inner = np.random.uniform(0.3, 0.48, N_valid_interface)
x_valid_inner = r_valid_inner * np.cos(theta_valid)
y_valid_inner = r_valid_inner * np.sin(theta_valid)
z_valid_inner = sign_x(x_valid_inner.reshape(-1, 1), y_valid_inner.reshape(-1, 1))
X_valid_interface_inner = np.column_stack([x_valid_inner, y_valid_inner, z_valid_inner.flatten()])

# Outer ring verification point (r = 0.52 to 0.7)  
r_valid_outer = np.random.uniform(0.52, 0.7, N_valid_interface)
x_valid_outer = r_valid_outer * np.cos(theta_valid)
y_valid_outer = r_valid_outer * np.sin(theta_valid)
z_valid_outer = sign_x(x_valid_outer.reshape(-1, 1), y_valid_outer.reshape(-1, 1))
X_valid_interface_outer = np.column_stack([x_valid_outer, y_valid_outer, z_valid_outer.flatten()])

# 3. Combine all validation points
X_valid_inner_combined = np.vstack([
    X_valid_inner_random, 
    X_valid_interface_inner, 
    X_valid_interface_outer
])

# Compute weights and source terms for interface validation points
# Inner circle weights
weights_interface_inner = np.ones(N_valid_interface) * (4.0 / (2 * N_valid_interface))
x_inner = X_valid_interface_inner[:, 0:1]
y_inner = X_valid_interface_inner[:, 1:2] 
z_inner = X_valid_interface_inner[:, 2:3]
Rf_interface_inner = rF(x_inner, y_inner, z_inner)

# Outer ring weights
weights_interface_outer = np.ones(N_valid_interface) * (4.0 / (2 * N_valid_interface))
x_outer = X_valid_interface_outer[:, 0:1]
y_outer = X_valid_interface_outer[:, 1:2] 
z_outer = X_valid_interface_outer[:, 2:3]
Rf_interface_outer = rF(x_outer, y_outer, z_outer)

# Combine weights and source terms
weights_valid_inner_combined = np.hstack([
    weights_valid_inner_random, 
    weights_interface_inner,
    weights_interface_outer
])

Rf_valid_inner_combined = np.vstack([
    Rf_valid_inner_random,
    Rf_interface_inner,
    Rf_interface_outer
])

# Switch validation variables to torch
X_valid_inner_torch = torch.tensor(X_valid_inner_combined, requires_grad=True, device=device, dtype=torch.float64)
weights_valid_inner_torch = torch.tensor(weights_valid_inner_combined).double().to(device)
Rf_valid_inner_torch = torch.tensor(Rf_valid_inner_combined, device=device, dtype=torch.float64)

# Improved validation points visualization
plt.figure(figsize=(12, 5))

# Left plot: Original training points distribution
plt.subplot(1, 2, 1)
plt.scatter(X_inner[:, 0], X_inner[:, 1], c="lightblue", s=0.5, marker=".", alpha=0.6, label="Inner training")
plt.scatter(X_bd[:, 0], X_bd[:, 1], c="red", s=3, marker=".", label="Boundary")
plt.scatter(X_interface[:, 0], X_interface[:, 1], c="black", s=3, marker="o", label="Interface")
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.title('Training Points Distribution')
plt.legend()

# Right plot: Improved validation points distribution
plt.subplot(1, 2, 2)
# Random inner validation points
plt.scatter(X_valid_inner_random[:, 0], X_valid_inner_random[:, 1], 
           c="blue", s=8, marker=".", alpha=0.7, label="Random inner")
# Interface inner circle verification point
plt.scatter(X_valid_interface_inner[:, 0], X_valid_interface_inner[:, 1], 
           c="orange", s=15, marker="^", alpha=0.8, label="Interface inner")
# Interface outer ring verification point
plt.scatter(X_valid_interface_outer[:, 0], X_valid_interface_outer[:, 1], 
           c="red", s=15, marker="s", alpha=0.8, label="Interface outer")
# Draw the interface circles as a reference
theta_ref = np.linspace(0, 2*np.pi, 100)
plt.plot(r0 * np.cos(theta_ref), r0 * np.sin(theta_ref), 'k--', linewidth=2, alpha=0.5, label=f"Interface (r={r0})")

plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.title('Improved Validation Points')
plt.legend()

plt.tight_layout()
plt.savefig('results_DeepNitsche/Improved_validation_points.png', dpi=300)
plt.show()
plt.close()


# Define the model
# Set the number of neurons in the hidden layer
num_neurons = 40 # 40, 60, 80, 100
model = Plain(3, num_neurons, 1).to(device)

# Set initial value
LBFGS_iter = 2000 # 1500, 2000
itera = 0
savedloss = []

model.train()

# Set the L-BFGS optimizer
optimizerLBFGS = torch.optim.LBFGS(
    model.parameters(),
    lr=1e-1,
    max_iter=20,
    tolerance_grad=1e-5,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn="strong_wolfe"
)

# Define closure function
def closure():
    optimizerLBFGS.zero_grad()
    

    lossLBFGS = loss(model, X_inner_torch, weights_inner_torch, Rf_inner_torch, X_bd_torch, U_bd_torch, X_interface_torch, Normal_interface_torch, Jump_torch, Jump_normal_torch, N_inner)
    # lossLBFGS_valid = loss(model, X_valid_inner_torch, weights_valid_inner_torch, Rf_valid_inner_torch, X_bd_torch,
    #                       U_bd_torch,
    #                       X_interface_torch, Normal_interface_torch, Jump_torch, Jump_normal_torch, N_inner)

    savedloss.append(lossLBFGS.item())
    lossLBFGS.backward(retain_graph=True)
    return lossLBFGS

# Independently verify weights outside the training loop
print("=== Pre-training Verification ===")
weights_sum = torch.sum(weights_inner_torch)
print(f"Weights sum: {weights_sum:.6f} (target: 4.0)")
print(f"Model parameters: {count_parameters(model)}")

# Start time
start_time = time.time()

# Training loop
while itera < LBFGS_iter:
    optimizerLBFGS.step(closure)

    if itera % 50 == 0 or itera == 0:
        print(f"Iter {itera}, LossLBFGS: {savedloss[-1]:.5e}")

    itera += 1

# Training completed
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")
print(f"Final loss: {savedloss[-1]:.5e}")

# L2 and H1 error analysis
model.eval()
# error_results = complete_error_analysis() 


# Plot the training and validation loss (improved visualization)
fig = plt.figure(figsize=(12, 5))

# 左图：绝对值Loss趋势
plt.subplot(1, 2, 1)
idx = list(range(len(savedloss)))
abs_loss = [abs(loss) for loss in savedloss]
plt.plot(idx, abs_loss, label="Training Loss (|L|)", linewidth=2, color='blue')
plt.xlabel("Iterations")
plt.ylabel("Loss (Absolute Value)")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Training Loss Convergence")

# 中图：损失变化率（收敛指标）
plt.subplot(1, 2, 2)
if len(savedloss) > 1:
    loss_changes = [abs(savedloss[i] - savedloss[i-1]) for i in range(1, len(savedloss))]
    plt.plot(range(1, len(savedloss)), loss_changes, color='red', linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Loss Change Rate")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.title("Loss Convergence Rate")

plt.tight_layout()
plt.savefig('results_DeepNitsche/Training_Loss_Comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Error detacting
# number of test points
# Error detection
N_test = 10000

# Generate test points using Latin Hypercube Sampling
X_inn = 2.0 * lhs(2, N_test) - 1.0
xx = X_inn[:, 0]
yy = X_inn[:, 1]
zz = sign_x(xx.reshape(-1, 1), yy.reshape(-1, 1))
u_test = exact_u(xx.reshape(-1, 1), yy.reshape(-1, 1), zz).flatten()
np.savetxt('results_DeepNitsche/numerical_solution/exact_solution/x_test.txt', xx.reshape(-1, 1), fmt='%.8e')
np.savetxt('results_DeepNitsche/numerical_solution/exact_solution/y_test.txt', yy.reshape(-1, 1), fmt='%.8e')
np.savetxt('results_DeepNitsche/numerical_solution/exact_solution/u_test.txt', u_test.reshape(-1, 1), fmt='%.8e')
X_inn_3d = np.column_stack([xx, yy, zz.flatten()])
X_inn_torch = torch.tensor(X_inn_3d, device=device).double()

# Predict using trained model
with torch.no_grad():
    u_pred = model(X_inn_torch).cpu().numpy().flatten()

# Verify shapes before plotting
print(f"Array shapes for plotting:")
print(f"xx: {xx.shape}")
print(f"yy: {yy.shape}")
print(f"u_pred: {u_pred.shape}")
print(f"u_test: {u_test.shape}")
print(f"u_pred type: {u_pred.dtype}")
print(f"u_test type: {u_test.dtype}")
np.savetxt('results_DeepNitsche/numerical_solution/pred_solution/x_pred.txt', xx.reshape(-1, 1), fmt='%.8e')
np.savetxt('results_DeepNitsche/numerical_solution/pred_solution/y_pred.txt', yy.reshape(-1, 1), fmt='%.8e')
np.savetxt('results_DeepNitsche/numerical_solution/pred_solution/u_pred.txt', u_pred.reshape(-1, 1), fmt='%.8e')

# Calculate errors
error = np.absolute(u_pred - u_test)

# Error metrics
error_u_inf_r = np.linalg.norm(error, np.inf) / np.linalg.norm(u_test, np.inf)
error_u_2r = np.linalg.norm(error, 2) / np.linalg.norm(u_test, 2)
error_u_inf = np.linalg.norm(error, np.inf)
error_u_2 = np.linalg.norm(error, 2) / np.sqrt(N_test)

# Convergence Analysis
print('=== Loss Convergence Analysis ===')
abs_loss = [abs(loss) for loss in savedloss]
print(f'Initial Loss: {abs_loss[0]:.5e}')
print(f'Final Loss: {abs_loss[-1]:.5e}')
print(f'Loss Reduction: {(abs_loss[0] - abs_loss[-1]) / abs_loss[0] * 100:.2f}%')

# Check overall convergence stability
if len(abs_loss) >= 50:
    # Calculate overall stability based on final convergence
    final_portion = abs_loss[-min(50, len(abs_loss)//4):]  # Last 25% or 50 iterations
    std_dev = np.std(final_portion)
    mean_val = np.mean(final_portion)
    cv = std_dev / mean_val * 100
    print(f'Final convergence stability - CV: {cv:.6f}% (lower is more stable)')

    if cv < 0.01:
        print('✅ Loss is highly stable (CV < 0.01%)')
    elif cv < 0.1:
        print('✅ Loss is stable (CV < 0.1%)')
    else:
        print('⚠️ Loss may still be converging')

# Find approximate convergence point
if len(savedloss) > 50:
    changes = [abs(savedloss[i] - savedloss[i-1]) for i in range(1, len(savedloss))]
    avg_change = np.mean(changes)
    threshold = avg_change * 0.01  # 1% of average change

    convergence_point = None
    for i in range(50, len(changes)):
        if all(change < threshold for change in changes[i-10:i]):
            convergence_point = i
            break

    if convergence_point:
        print(f'📍 Approximate convergence achieved around iteration: {convergence_point}')
    else:
        print('📍 Training shows good convergence trend')

print()  

print('=== Error Analysis ===')
print(f'Error u (relative inf-norm): {error_u_inf_r:.5e}')
print(f'Error u (relative 2-norm): {error_u_2r:.5e}')
print(f'Error u (absolute inf-norm): {error_u_inf:.5e}')
print(f'Error u (absolute 2-norm): {error_u_2:.5e}')

# Model 3D plot
# 3D comparison plot
fig = plt.figure(figsize=(15, 6))

# Predicted solution
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.scatter(xx, yy, u_pred, c=u_pred, cmap='viridis', s=1, alpha=0.6)
fig.colorbar(surf1, shrink=0.5, aspect=5)
ax1.set_title('Predicted Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')

# Exact solution
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.scatter(xx, yy, u_test, c=u_test, cmap='rainbow', s=1, alpha=0.6)
# surf2 = ax2.scatter(xx, yy, u_test, c=u_test, cmap='plasma', s=1, alpha=0.6)
fig.colorbar(surf2, shrink=0.5, aspect=5)
ax2.set_title('Exact Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u')
plt.tight_layout()
plt.savefig('results_DeepNitsche/Model_3D_Predicted_vs_Exact.png', dpi=300)
plt.show()
plt.close()


# Error visualization
fig = plt.figure(figsize=(15, 6))

# 3D error plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.scatter(xx, yy, error, c=error, cmap='viridis', marker=".", s=5)
fig.colorbar(surf1, shrink=0.5, aspect=5)
ax1.view_init(elev=60, azim=10)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Error')
ax1.set_title('Absolute Error (3D)')

# 2D error plot
ax2 = fig.add_subplot(1, 2, 2)
surf2 = ax2.scatter(xx, yy, c=error, cmap='OrRd', s=5)
fig.colorbar(surf2, shrink=0.5, aspect=5)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Absolute Error (2D)')
ax2.axis('equal')

# Mark the interface on the 2D plot
theta_circle = np.linspace(0, 2*np.pi, 100)
x_circle = r0 * np.cos(theta_circle)
y_circle = r0 * np.sin(theta_circle)
ax2.plot(x_circle, y_circle, 'k--', linewidth=2, label='Interface')
ax2.legend()
plt.tight_layout()
plt.savefig('results_DeepNitsche/Error_Visualization.png', dpi=300)
plt.show()
plt.close()
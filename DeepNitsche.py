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
    level_sets = [L1(x, y), L2(x, y), L3(x, y)]
    for idx, li in enumerate(level_sets):
        mask = li < 0
        z[mask] = -(idx + 1)
    return z


set_seed(42)
# beta define
# (beta_plus, beta_minus)
beta_plus = 1.0 # Liang
beta_minus = 10.0
alpha = 1.0
r0 = 0.5

# Three-circle level-set parameters
r1 = 0.25
r2 = 0.30
r3 = 0.20
circle_centers = [
    (0.5, 0.5),   # Circle 1
    (-0.5, 0.5),  # Circle 2
    (0.5, -0.5),  # Circle 3
]
alpha_1 = 1.0
alpha_2 = 1.0
alpha_3 = 1.0
radii = [r1, r2, r3]
alpha_list = [alpha_1, alpha_2, alpha_3]
PROJECTION_EPS = 1e-12


def level_set(x, y, cx, cy, radius):
    return (x - cx) ** 2 + (y - cy) ** 2 - radius ** 2


def L1(x, y):
    return level_set(x, y, circle_centers[0][0], circle_centers[0][1], r1)


def L2(x, y):
    return level_set(x, y, circle_centers[1][0], circle_centers[1][1], r2)


def L3(x, y):
    return level_set(x, y, circle_centers[2][0], circle_centers[2][1], r3)


def combined_L(x, y):
    return L1(x, y) * L2(x, y) * L3(x, y)


def _compute_jump_constants(samples=720):
    constants = []
    theta = torch.linspace(0.0, 2.0 * np.pi, samples, device=device, dtype=torch.float64)
    for idx in range(3):
        radius = radii[idx]
        cx, cy = circle_centers[idx]
        x_curve = cx + radius * torch.cos(theta)
        y_curve = cy + radius * torch.sin(theta)
        x_curve = x_curve.clone().detach().requires_grad_(True)
        y_curve = y_curve.clone().detach().requires_grad_(True)
        L_val = combined_L(x_curve, y_curve)
        gradLx = torch.autograd.grad(
            L_val, x_curve,
            grad_outputs=torch.ones_like(L_val),
            create_graph=True,
            retain_graph=True
        )[0]
        gradLy = torch.autograd.grad(
            L_val, y_curve,
            grad_outputs=torch.ones_like(L_val),
            create_graph=True,
            retain_graph=True
        )[0]
        nx = (x_curve - cx) / radius
        ny = (y_curve - cy) / radius
        normal_dot = gradLx * nx + gradLy * ny
        constants.append(alpha_list[idx] * normal_dot.mean().item())
        x_curve.detach_()
        y_curve.detach_()
    return constants


jump_constants = None


def _exact_u_torch(x, y):
    L_total = combined_L(x, y)
    level_sets = [L1(x, y), L2(x, y), L3(x, y)]
    u_val = L_total

    for idx, li in enumerate(level_sets):
        mask = li < 0
        if torch.any(mask):
            jump = torch.tensor(jump_constants[idx], dtype=x.dtype, device=x.device)
            u_val = torch.where(mask, L_total - jump, u_val)

    return u_val


def exact_u(x, y, z):
    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float64, device=device)
    u_tensor = _exact_u_torch(x_tensor, y_tensor)
    return u_tensor.detach().cpu().numpy()

# the gradient of the exact_u
def exact_du(x, y, z):
    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device).clone().detach().requires_grad_(True)
    y_tensor = torch.as_tensor(y, dtype=torch.float64, device=device).clone().detach().requires_grad_(True)
    u_tensor = _exact_u_torch(x_tensor, y_tensor)
    grad_outputs = torch.ones_like(u_tensor, dtype=torch.float64, device=device)
    dux, duy = torch.autograd.grad(
        u_tensor,
        (x_tensor, y_tensor),
        grad_outputs=grad_outputs,
        retain_graph=False,
        create_graph=False
    )
    return dux.detach().cpu().numpy(), duy.detach().cpu().numpy()

# V define. [u] jump across interfaces
def V(x_interface, y_interface):
    x_tensor = torch.as_tensor(x_interface, dtype=torch.float64, device=device)
    y_tensor = torch.as_tensor(y_interface, dtype=torch.float64, device=device)

    level_values = torch.stack([
        torch.abs(L1(x_tensor, y_tensor)),
        torch.abs(L2(x_tensor, y_tensor)),
        torch.abs(L3(x_tensor, y_tensor))
    ], dim=0)

    closest_idx = torch.argmin(level_values, dim=0)
    jump_tensor = torch.zeros_like(x_tensor)

    for idx in range(3):
        mask = (closest_idx == idx)
        if torch.any(mask):
            jump_tensor[mask] = jump_constants[idx]

    return jump_tensor.detach().cpu().numpy()


def W(x_interface, y_interface):
    """
    三个界面的法向通量跳跃为零
    """
    return np.zeros_like(x_interface)


def laplacian_L(x, y):
    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device).clone().detach().requires_grad_(True)
    y_tensor = torch.as_tensor(y, dtype=torch.float64, device=device).clone().detach().requires_grad_(True)
    L_val = combined_L(x_tensor, y_tensor)
    gradLx = torch.autograd.grad(
        L_val, x_tensor,
        grad_outputs=torch.ones_like(L_val),
        create_graph=True,
        retain_graph=True
    )[0]
    gradLy = torch.autograd.grad(
        L_val, y_tensor,
        grad_outputs=torch.ones_like(L_val),
        create_graph=True,
        retain_graph=True
    )[0]
    d2x = torch.autograd.grad(
        gradLx, x_tensor,
        grad_outputs=torch.ones_like(gradLx),
        retain_graph=True
    )[0]
    d2y = torch.autograd.grad(
        gradLy, y_tensor,
        grad_outputs=torch.ones_like(gradLy),
        retain_graph=True
    )[0]
    lap = d2x + d2y
    return lap.detach().cpu().numpy()


def rF(x, y, z):
    lap = laplacian_L(x, y)
    return -lap

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
jump_constants = _compute_jump_constants()

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
def loss(model, X_inner, weights, Rf_inner, X_bd, U_bd, X_interface, Normal_interface, Jump, Jump_normal, Interface_labels, Interface_ds, N_inner):
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
    interface_outer = torch.cat([
        X_interface[:, 0:2],
        torch.ones(X_interface.shape[0], 1, device=X_interface.device, dtype=X_interface.dtype, requires_grad=True)
    ], dim=1)
    interface_inner = torch.cat([X_interface[:, 0:2], Interface_labels], dim=1)
    interface_outer.requires_grad_(True)
    interface_inner.requires_grad_(True)

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
    
    gamma = 1.0 
    line_integral = gamma / alpha * torch.sum((jump_pred.squeeze()) ** 2 * Interface_ds.squeeze())

    

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
quarant_number_one_side = 24 # 16, 24, 32, 48, 64

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

def sample_circle_interface(radius, center, n_points):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    normals = normal_vector(x, y)
    jumps = V(x, y)
    jump_normals = W(x, y)
    arc_length = 2 * np.pi * radius
    return x, y, normals, jumps, jump_normals, arc_length


points_per_circle = np.full(3, N_interface // 3, dtype=int)
points_per_circle[:N_interface % 3] += 1

interface_data = []
total_arc_length = 0.0
interface_labels = []
interface_ds = []

for idx in range(3):
    x, y, normals, jumps, jump_normals, arc_length = sample_circle_interface(
        radii[idx],
        circle_centers[idx],
        points_per_circle[idx]
    )
    total_arc_length += arc_length
    interface_data.append({
        "x": x,
        "y": y,
        "normals": normals,
        "jump": jumps,
        "jump_normal": jump_normals,
        "arc_length": arc_length,
        "ds": arc_length / points_per_circle[idx]
    })
    interface_labels.append(np.full(points_per_circle[idx], -(idx + 1.0)))
    interface_ds.append(np.full(points_per_circle[idx], arc_length / points_per_circle[idx]))

x_interface = np.concatenate([data["x"] for data in interface_data])
y_interface = np.concatenate([data["y"] for data in interface_data])
X_interface = np.vstack((x_interface, y_interface)).T

Normal_interface = np.vstack([data["normals"] for data in interface_data])
Jump = np.concatenate([data["jump"] for data in interface_data])
Jump_normal = np.concatenate([data["jump_normal"] for data in interface_data])
interface_labels = np.concatenate(interface_labels)
interface_ds = np.concatenate(interface_ds)

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
Interface_labels_torch = torch.tensor(interface_labels, device=device, dtype=torch.float64).unsqueeze(1)
Interface_ds_torch = torch.tensor(interface_ds, device=device, dtype=torch.float64).unsqueeze(1)

# Training points plot
plt.figure(figsize=(5, 5))

plt.scatter(X_inner[:, 0], X_inner[:, 1],
            c="b", s=1, marker=".")
plt.scatter(X_bd[:, 0], X_bd[:, 1],
            c="r", s=5, marker=".")
plt.scatter(X_interface[:, 0], X_interface[:, 1],
            c="k", s=5, marker="o")
for idx, radius in enumerate(radii):
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = circle_centers[idx]
    plt.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), linestyle="--", linewidth=1.0)
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

# 2. Structured validation points near each interface
np.random.seed(123)  # Ensure reproducibility
points_per_circle_valid = np.full(3, max(1, N_valid_interface // 3), dtype=int)
points_per_circle_valid[:N_valid_interface % 3] += 1

inner_points = []
outer_points = []

for idx in range(3):
    n_pts = points_per_circle_valid[idx]
    theta_valid = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    # inner ring: slightly inside each circle
    r_inner = np.random.uniform(0.85 * radii[idx], 0.98 * radii[idx], n_pts)
    # outer ring: slightly outside each circle
    r_outer = np.random.uniform(1.02 * radii[idx], 1.25 * radii[idx], n_pts)
    cx, cy = circle_centers[idx]

    x_inner = cx + r_inner * np.cos(theta_valid)
    y_inner = cy + r_inner * np.sin(theta_valid)
    z_inner = sign_x(x_inner.reshape(-1, 1), y_inner.reshape(-1, 1))
    inner_points.append(np.column_stack([x_inner, y_inner, z_inner.flatten()]))

    x_outer = cx + r_outer * np.cos(theta_valid)
    y_outer = cy + r_outer * np.sin(theta_valid)
    z_outer = sign_x(x_outer.reshape(-1, 1), y_outer.reshape(-1, 1))
    outer_points.append(np.column_stack([x_outer, y_outer, z_outer.flatten()]))

if inner_points:
    X_valid_interface_inner = np.vstack(inner_points)
    X_valid_interface_outer = np.vstack(outer_points)
else:
    X_valid_interface_inner = np.empty((0, 3))
    X_valid_interface_outer = np.empty((0, 3))

# 3. Combine all validation points
X_valid_inner_combined = np.vstack([
    X_valid_inner_random,
    X_valid_interface_inner,
    X_valid_interface_outer
])

# Compute weights and source terms for interface validation points
num_inner = X_valid_interface_inner.shape[0]
num_outer = X_valid_interface_outer.shape[0]

weights_interface_inner = np.ones(num_inner) * (4.0 / (2 * max(1, num_inner)))
x_inner = X_valid_interface_inner[:, 0:1] if num_inner > 0 else np.empty((0, 1))
y_inner = X_valid_interface_inner[:, 1:2] if num_inner > 0 else np.empty((0, 1))
z_inner = X_valid_interface_inner[:, 2:3] if num_inner > 0 else np.empty((0, 1))
Rf_interface_inner = rF(x_inner, y_inner, z_inner) if num_inner > 0 else np.empty((0, 1))

weights_interface_outer = np.ones(num_outer) * (4.0 / (2 * max(1, num_outer)))
x_outer = X_valid_interface_outer[:, 0:1] if num_outer > 0 else np.empty((0, 1))
y_outer = X_valid_interface_outer[:, 1:2] if num_outer > 0 else np.empty((0, 1))
z_outer = X_valid_interface_outer[:, 2:3] if num_outer > 0 else np.empty((0, 1))
Rf_interface_outer = rF(x_outer, y_outer, z_outer) if num_outer > 0 else np.empty((0, 1))

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
theta_ref = np.linspace(0, 2 * np.pi, 200)
for idx, radius in enumerate(radii):
    cx, cy = circle_centers[idx]
    plt.plot(cx + radius * np.cos(theta_ref), cy + radius * np.sin(theta_ref), 'k--', linewidth=1.0, alpha=0.7)
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
if X_valid_interface_inner.size > 0:
    plt.scatter(X_valid_interface_inner[:, 0], X_valid_interface_inner[:, 1], 
               c="orange", s=15, marker="^", alpha=0.8, label="Interface inner")
# Interface outer ring verification point
if X_valid_interface_outer.size > 0:
    plt.scatter(X_valid_interface_outer[:, 0], X_valid_interface_outer[:, 1], 
               c="red", s=15, marker="s", alpha=0.8, label="Interface outer")
# Draw the interface circles as a reference
theta_ref = np.linspace(0, 2*np.pi, 200)
for idx, radius in enumerate(radii):
    cx, cy = circle_centers[idx]
    plt.plot(cx + radius * np.cos(theta_ref), cy + radius * np.sin(theta_ref), 'k--', linewidth=1.5, alpha=0.7)

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
LBFGS_iter = 500 # 1500, 2000
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
    

    lossLBFGS = loss(model, X_inner_torch, weights_inner_torch, Rf_inner_torch, X_bd_torch, U_bd_torch, X_interface_torch, Normal_interface_torch, Jump_torch, Jump_normal_torch, Interface_labels_torch, Interface_ds_torch, N_inner)
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
# error_results = complete_error_analysis(
#     model,
#     X_inner_torch,
#     X_inner,
#     weights_inner,
#     beta_plus,
#     beta_minus,
#     exact_u,
#     exact_du
# ) 


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
theta_circle = np.linspace(0, 2*np.pi, 200)
for idx, radius in enumerate(radii):
    cx, cy = circle_centers[idx]
    x_circle = cx + radius * np.cos(theta_circle)
    y_circle = cy + radius * np.sin(theta_circle)
    ax2.plot(x_circle, y_circle, 'k--', linewidth=2, label=f'Interface {idx+1}')
ax2.legend()
plt.tight_layout()
plt.savefig('results_DeepNitsche/Error_Visualization.png', dpi=300)
plt.show()
plt.close()

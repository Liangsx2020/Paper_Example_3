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

# Count parameters

def count_parameters(model, requires_grad=True):
    """Count trainable parameters for a nn.Module."""
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def compute_errors_paper_format(model,
                                X_inner_torch,
                                X_inner_np,
                                weights_inner_np,
                                beta_plus,
                                beta_minus,
                                exact_u_fn,
                                exact_du_fn):
    """
    计算论文格式的L²和H¹离散误差
    """
    # 重新创建需要梯度的输入张量
    X_inner_grad = X_inner_torch.clone().detach().requires_grad_(True)
    
    # 预测解和梯度
    with torch.no_grad():
        u_pred_torch = model(X_inner_grad)
    
    # 需要重新设置requires_grad来计算梯度
    X_inner_grad = X_inner_torch.clone().detach().requires_grad_(True)
    u_pred_torch = model(X_inner_grad)
    u_pred = u_pred_torch.cpu().detach().numpy().flatten()
    
    # 计算梯度
    grad_u_pred = torch.autograd.grad(
        u_pred_torch, X_inner_grad,
        grad_outputs=torch.ones_like(u_pred_torch),
        create_graph=False,
        retain_graph=False
    )[0]
    
    ux_pred = grad_u_pred[:, 0].cpu().detach().numpy()
    uy_pred = grad_u_pred[:, 1].cpu().detach().numpy()
    
    # 精确解
    x_inner = X_inner_np[:, 0:1]
    y_inner = X_inner_np[:, 1:2]
    z_inner = X_inner_np[:, 2:3]
    u_exact = exact_u_fn(x_inner, y_inner, z_inner).flatten()
    ux_exact, uy_exact = exact_du_fn(x_inner, y_inner, z_inner)
    
    # 权重 (应该和为4)
    weights_np = weights_inner_np.flatten()
    
    # L²误差
    u_diff = u_pred - u_exact
    L2_error_squared = np.dot(weights_np, u_diff**2)
    L2_error = np.sqrt(L2_error_squared)
    
    # 离散H¹误差（包含分片常数β）
    z_values = z_inner.flatten()
    beta_values = np.where(z_values > 0, beta_plus, beta_minus)  # 根据区域确定β值
    
    # 梯度误差（加权）
    ux_diff = ux_pred - ux_exact.flatten()
    uy_diff = uy_pred - uy_exact.flatten()
    
    grad_error_weighted = beta_values * (ux_diff**2 + uy_diff**2)
    grad_norm_squared = np.dot(weights_np, grad_error_weighted)
    
    # 离散H¹范数
    H1_discrete_error = np.sqrt(L2_error_squared + grad_norm_squared)
    
    print(f"‖u - uₕ‖_{{L²(Ω)}}: {L2_error:.3e}")
    print(f"‖u - uₕ‖_{{1,h}}: {H1_discrete_error:.3e}")
    
    return L2_error, H1_discrete_error

def complete_error_analysis(model,
                            X_inner_torch,
                            X_inner_np,
                            weights_inner_np,
                            beta_plus,
                            beta_minus,
                            exact_u_fn,
                            exact_du_fn):
    """
    完整的误差分析，包括绝对和相对误差
    """
    print("=== Complete Error Analysis ===")
    
    # 计算误差
    L2_abs, H1_abs = compute_errors_paper_format(
        model,
        X_inner_torch,
        X_inner_np,
        weights_inner_np,
        beta_plus,
        beta_minus,
        exact_u_fn,
        exact_du_fn
    )
    
    # 计算精确解的范数（用于相对误差）
    x_inner = X_inner_np[:, 0:1]
    y_inner = X_inner_np[:, 1:2]
    z_inner = X_inner_np[:, 2:3]
    u_exact = exact_u_fn(x_inner, y_inner, z_inner).flatten()
    ux_exact, uy_exact = exact_du_fn(x_inner, y_inner, z_inner)
    
    weights_np = weights_inner_np.flatten()
    
    # 精确解的L²范数
    u_exact_L2_squared = np.dot(weights_np, u_exact**2)
    u_exact_L2 = np.sqrt(u_exact_L2_squared)
    
    # 精确解的H¹范数
    z_values = z_inner.flatten()
    beta_values = np.where(z_values > 0, beta_plus, beta_minus)
    grad_exact_weighted = beta_values * (ux_exact.flatten()**2 + uy_exact.flatten()**2)
    grad_exact_norm_squared = np.dot(weights_np, grad_exact_weighted)
    u_exact_H1 = np.sqrt(u_exact_L2_squared + grad_exact_norm_squared)
    
    # 相对误差
    L2_relative = L2_abs / u_exact_L2
    H1_relative = H1_abs / u_exact_H1
    
    print(f"\nAbsolute Errors:")
    print(f"L² error: {L2_abs:.5e}")
    print(f"H¹ error: {H1_abs:.5e}")
    
    print(f"\nRelative Errors:")
    print(f"L² relative error: {L2_relative:.5e}")
    print(f"H¹ relative error: {H1_relative:.5e}")
    
    print(f"\nExact Solution Norms:")
    print(f"‖u‖_{{L²}}: {u_exact_L2:.5e}")
    print(f"‖u‖_{{H¹}}: {u_exact_H1:.5e}")
    
    return {
        'L2_abs': L2_abs,
        'H1_abs': H1_abs, 
        'L2_rel': L2_relative,
        'H1_rel': H1_relative,
        'exact_L2_norm': u_exact_L2,
        'exact_H1_norm': u_exact_H1
    }

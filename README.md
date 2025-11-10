# DeepNitsche_ex3

基于 Nitsche 方法的物理信息神经网络（PINN）示例，用于求解三重界面椭圆型界面问题。项目实现了精确解构造、Gauss-Legendre 采样、界面跳跃/通量约束以及基于 L-BFGS 的训练流程，并提供了训练日志、误差分析与可视化产物的完整链条。

## 功能亮点
- **多界面显式建模**：`DeepNitsche.py` 在单位方形内布置三条圆形界面，显式编码解的跳跃常数与法向通量连续条件。
- **高精度采样与积分**：内部/边界点基于 Gauss-Legendre 节点生成，支持 `quarant_number_one_side` 控制精度；同时保存权重以计算 L²/H¹ 离散误差。
- **Nitsche 型损失**：损失函数包含体积分、界面跳跃、边界一致性与罚项，可通过 `eta`、`beta_plus/minus`、`gamma` 等参数调节稳定性。
- **可重复实验输出**：训练曲线、误差可视化与预测/精确解对比图统一写入 `results_DeepNitsche/`；长时任务可用 `log_DeepNitsche.py` 后台运行并将 stdout 重定向至 `logs/`。
- **误差后处理**：`utils.py` 提供 `complete_error_analysis()`，可在训练结束后输出论文格式的 L²/H¹ 绝对与相对误差。

## 代码结构
```
DeepNitsche.py        # 主训练脚本：精确解/采样/损失/训练与可视化
log_DeepNitsche.py    # nohup 启动器，示例日志写入 logs/debug_*.log
utils.py              # 参数统计与 L²/H¹ 误差分析函数
three_circle.md       # 三圆界面问题的数学描述
results_DeepNitsche/  # 训练生成的图像与预测结果
logs/                 # 长时间训练的日志（需手动命名）
```

## 环境准备
1. **Python**：建议 3.9+，确保默认浮点为 `torch.float64`（脚本已显式 `.double()`）。
2. **依赖**：
   ```bash
   pip install torch numpy matplotlib scipy pyDOE
   ```
   若使用 GPU，请安装与 CUDA 匹配的 PyTorch 版本。
3. （可选）创建虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

## 快速开始
1. **前台训练**
   ```bash
   python DeepNitsche.py
   ```
   - 默认使用 `quarant_number_one_side=128`、`num_neurons=40`、L-BFGS 迭代 500 次。
   - 训练过程中会打印权重求和校验与损失值，并在 `results_DeepNitsche/` 生成采样分布、训练曲线与误差可视化。
2. **后台训练**
   ```bash
   python log_DeepNitsche.py
   ```
   - 请根据超参数修改 `log_DeepNitsche.py` 中的日志文件名，例如 `logs/m_128_N_40_eta_50000.log`，保持命名能反映实验配置。

## 训练输出
运行结束后可在 `results_DeepNitsche/` 找到：
- `Training_Loss_Comprehensive.png`：L-BFGS 损失趋势（线性/对数刻度）。
- `Error_Visualization.png`：数值解与精确解误差对比。
- `Model_3D_Predicted_vs_Exact.png`：3D 表面图展示预测/解析解。
- `numerical_solution/pred_solution/`：栅格化数值解文件，方便后续分析。
- `Improved_validation_points.png`：训练与验证采样点分布。

若修改了超参数（`quarant_number_one_side`、`num_neurons`、`LBFGS_iter` 等），请同步调整结果子目录或文件名前缀，例如 `results_DeepNitsche/m128_N40_eta50000/`，便于多实验对比。

## 误差评估
训练脚本默认不自动运行误差分析。完成训练后进入 Python 解释器或执行：
```bash
python - <<'PY'
from utils import complete_error_analysis
from DeepNitsche import (model, X_inner_torch, X_inner, weights_inner,
                         beta_plus, beta_minus, exact_u, exact_du)
complete_error_analysis(model, X_inner_torch, X_inner, weights_inner,
                        beta_plus, beta_minus, exact_u, exact_du)
PY
```
该函数会打印：
- 绝对误差：‖u - uₕ‖_{L²(Ω)}、‖u - uₕ‖_{1,h}
- 相对误差
- 精确解的 L²/H¹ 范数（用于报告）

## 常用配置项
- `quarant_number_one_side`：Gauss 节点数；增大以提高内点密度。
- `num_neurons`：`Plain` 网络的隐藏层宽度（40/60/80/100）。
- `LBFGS_iter`：外层 while 循环的迭代次数；较大值可提升收敛。
- `eta`、`gamma`、`alpha`、`beta_plus/minus`：Nitsche 罚项与介质参数。
- `solution_scale`/`r1~r3`/`circle_centers`：定义精确解与界面几何。

修改这些参数时，请遵循：
1. **随机种子**：`set_seed(42)` 保证相同采样；若需多次实验可在脚本顶部更新。
2. **输出命名**：与 AGENTS 指南一致，确保 `results_DeepNitsche/`、`logs/` 子目录名称包含关键超参数。
3. **日志/图像归档**：长时间实验请保留 log 与关键图像路径，方便 PR/报告引用。

## 验证流程
1. 观察终端损失是否单调下降，无异常震荡。
2. 检查 `results_DeepNitsche/Training_Loss_Comprehensive.png` 与 `Error_Visualization.png`。
3. 调用 `complete_error_analysis()` 记录 L²/H¹ 绝对/相对误差；如调整采样或损失项，请将新的误差和核心图像放入独立子目录（示例：`results_DeepNitsche/high_res/`）。

## 故障排查
- **损失发散**：调低 `eta` 或缩小 `solution_scale`，并确认 `jump_constants` 已根据新几何重新计算。
- **NaN 梯度**：确保输入张量 `requires_grad` 设置在需要的位置后及时 `.detach()`，并使用 `torch.float64`。
- **日志为空**：检查 `log_DeepNitsche.py` 中的输出路径是否存在，或使用 `nohup ... &` 前手动创建 `logs/`。

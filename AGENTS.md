# Repository Guidelines

## Project Structure & Module Organization
- `DeepNitsche.py` 是主实验脚本：定义精确解、采样点、Nitsche 型损失与训练/可视化流程。调整网格密度(`quarant_number_one_side`)、网络宽度(`num_neurons`)与迭代数时，务必同步更新图像或误差输出目录。
- `utils.py` 存放工具函数（参数计数、L²/H¹ 误差统计等）。它依赖主脚本在全局作用域中构造的张量，调用前请先完成训练并确保相关变量存在。
- `log_DeepNitsche.py` 通过 `nohup` 后台运行主脚本并把日志写入 `logs/`。默认输出名 `m_64_N_40_eta_2000_k+_10.log`，如修改超参数请更新文件名以避免覆盖。
- 结果与中间数据写入 `results_DeepNitsche/`（例如 `numerical_solution/pred_solution/`），版本化代码应保留目录结构但忽略体积大的 `.txt` 或图像产物，除非需要对比基准。

## Build, Test, and Development Commands
- `python DeepNitsche.py`：本地直接训练并在 `results_DeepNitsche/` 生成采样图、损失曲线与误差图。
- `python log_DeepNitsche.py`：后台训练，日志写入 `logs/*.log` 以便长跑任务。
- 训练完成后，可运行：
  ```bash
  python - <<'PY'
  from utils import complete_error_analysis; complete_error_analysis()
  PY
  ```
  以输出论文格式的 L²/H¹ 绝对与相对误差。

## Coding Style & Naming Conventions
- 统一采用 Python 3.9+、PEP 8 风格与 4 空格缩进；张量默认 `torch.float64`，不要混入 `float32` 以免损失强制连续条件。
- 新增函数优先放在 `DeepNitsche.py` 顶部或 `utils.py`，命名使用蛇形（如 `generate_validation_ring`）。绘图/输出路径以 `results_DeepNitsche/<task>` 命名，便于批量对比。
- 复杂张量操作可配少量行内注释，但保持“解释为什么而非如何”的语气。

## Testing Guidelines
- 项目未集成自动化测试；贡献者需手动验证训练收敛与误差。最小流程：`python DeepNitsche.py` → 检查终端损失、`results_DeepNitsche/Training_Loss_Comprehensive.png`、`Error_Visualization.png` → 运行 `complete_error_analysis()`。
- 若修改采样或损失项，请附带新的误差指标（L²、H¹、相对误差）及关键图像，命名格式如 `results_DeepNitsche/<variant>/...`。

## Commit & Pull Request Guidelines
- 当前提交历史仅含简短语句（例：`start to construct`）。建议沿用“祈使句 + 范围”模式（如 `refine loss weighting`），首字母小写，≤72 字符。
- PR 描述需包含：修改动机、核心改动要点、运行的命令及关键输出；若新增图像，请附缩略图或路径引用（例如 ``results_DeepNitsche/Model_3D_Predicted_vs_Exact.png``）。必要时链接相关 Issue，并说明是否影响复现实验或需要重新训练基准模型。

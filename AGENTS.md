# Repository Guidelines

## 项目结构与模块职责
`DeepNitsche.py` 为核心实验脚本，负责精确解定义、采样网格生成、Nitsche 型损失及训练/可视化流程；调整 `quarant_number_one_side`、`num_neurons` 或迭代数时，请同步更新输出目录命名。`utils.py` 存放辅助函数（参数统计、L²/H¹ 误差等），需在主脚本完成训练并构造相关张量后再调用。长时间任务使用 `log_DeepNitsche.py` 通过 `nohup` 后台运行，并根据超参数重命名日志文件（默认 `logs/m_64_N_40_eta_2000_k+_10.log`）。所有图像与预测结果置于 `results_DeepNitsche/`（如 `results_DeepNitsche/numerical_solution/pred_solution/`），保持目录结构以方便对比。

## 构建、测试与开发命令
本地训练：`python DeepNitsche.py`，运行结束后在 `results_DeepNitsche/` 自动生成采样图、损失曲线与误差可视化。后台训练：`python log_DeepNitsche.py`，日志写入 `logs/`，适合多小时运行。训练完成后执行
```bash
python - <<'PY'
from utils import complete_error_analysis; complete_error_analysis()
PY
```
即可输出论文格式的 L²/H¹ 绝对与相对误差。

## 编码风格与命名
统一使用 Python 3.9+、PEP 8、4 空格缩进。张量默认 `torch.float64`，避免混入 `float32` 以维持连续性约束。新增函数采用蛇形命名（如 `generate_validation_ring`），优先放在 `DeepNitsche.py` 顶部或 `utils.py`。输出路径沿用 `results_DeepNitsche/<task>` 模式，便于批量比较不同实验。

## 测试与验证
项目暂无自动化测试，需手动确认收敛：1）运行 `python DeepNitsche.py` 并观察终端损失；2）检查 `results_DeepNitsche/Training_Loss_Comprehensive.png` 与 `results_DeepNitsche/Error_Visualization.png`；3）调用 `complete_error_analysis()` 记录最终指标。若改动采样或损失项，请提供最新 L²/H¹ 绝对与相对误差及核心图像，并放入独立子目录（如 `results_DeepNitsche/high_res/`）。

## 提交与 PR 规范
提交信息应简短、祈使句、首字母小写（示例：`refine loss weighting`），长度控制在 72 字符内。PR 需说明动机、概述核心改动、列出运行命令及关键输出，并链接相关 Issue。若生成新图像，请给出路径（例如 ``results_DeepNitsche/Model_3D_Predicted_vs_Exact.png``）。最后注明是否影响可复现实验或需重新训练共享模型。

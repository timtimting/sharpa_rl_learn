# 2026-10-06 组合扰动鲁棒性训练记录

## 结论与归档

- 状态：训练 3000 万 agent steps 已完成，三组仿真评估已完成。
- 推荐模型：`artifacts/rotation_robust_2026-10-06/run/stage1_nn/best.pth`。相同评估条件下，它在两个扰动种子上的成功率均高于原模型和本次 `last.pth`。
- 独立归档：`artifacts/rotation_robust_2026-10-06/`，约 18 MiB，保留本次完整运行目录（模型、周期性快照、TensorBoard、配置、评估 JSON/日志）、训练控制台日志、原模型及关键源码快照。原始目录仍在 `logs/rotation_robust/2026-10-06_11-31-08/`。
- 注意：归档在本机，不能替代异机备份；仓库的 `.gitignore` 会忽略 `*.pth`，仅提交这份 Markdown **不会**保存模型权重。

## 训练目标与设置

- 任务：`Isaac-Inhand-Rotate-Sharpa-Wave-v0`；圆柱体在手内绕竖直轴旋转，同时避免因高度越界而提前结束。
- 初始策略：`logs/rotation_finetune/2026-09-25_23-24-22/stage1_nn/best.pth`，其副本为归档根目录的 `teacher_best.pth`。
- 固定重力：`[0, 0, -9.81] m/s²`，关闭重力课程；同时开启 PD 增益、质心、物体质量和摩擦系数随机化。
- PPO 继续训练：1024 个并行环境，训练种子 2026，`cuda:0`，学习率 `1e-4`，目标 30,000,000 agent steps；其余参数使用任务的 PPO 配置。
- 环境：Ubuntu 22.04.5、Isaac Sim 4.5、NVIDIA GeForce RTX 3060、驱动 580.178.04、PyTorch 2.7.0+cu128。仓库起点提交 `95ccda3`；训练时环境代码有未提交改动，相关文件已复制到归档的 `source/`。
- 启动脚本快照：`artifacts/rotation_robust_2026-10-06/source/rl_isaaclab/scripts/run_robust_finetune.sh`。其训练命令为：

```bash
python -u rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --num_envs 1024 --headless --device cuda:0 --seed 2026 \
  --resume --load_path logs/rotation_finetune/2026-09-25_23-24-22/stage1_nn/best.pth \
  --max_agent_steps 30000000 \
  'env.sim.gravity=[0.0,0.0,-9.81]' env.gravity_curriculum=false \
  env.randomize_pd_gains=true env.randomize_com=true \
  env.randomize_mass=true env.randomize_friction=true \
  agent.algorithm.learning_rate=0.0001 \
  agent.algorithm.experiment_name=rotation_robust
```

## 训练过程

| 阶段 | 记录 |
| --- | --- |
| 2026-10-06 11:31 左右 | 从原策略加载，开始鲁棒微调；初期回合奖励因扰动快速下降。 |
| 约 300 万步 | 训练回合平均奖励约 182；当时最佳约 379。 |
| 约 800 万步 | 训练回合平均奖励约 308；当时最佳约 449。 |
| 约 1500 万步 | TensorBoard 回合奖励约 348。 |
| 约 2400 万步 | 训练回合平均奖励约 349；当时最佳约 472。 |
| 29,425,664 步 | TensorBoard 记录本次最高训练回合平均奖励 480.83。 |
| 约 12:54 | 日志出现 `max steps achieved`；最终训练回合平均奖励约 389，训练采集耗时约 81.1 分钟。 |
| 13:09 前 | 扰动种子 42、123 及常规条件种子 42 的自动评估完成，生成 `summary.json`。 |

训练奖励有较大波动，不应直接当作旋转成功率。完整曲线在归档的 `run/stage1_tb/`；完整控制台记录在 `artifacts/rotation_robust_2026-10-06/train_console.log`。

## 评估方法

- 每组：64 个并行环境、1000 个仿真步、固定重力 −9.81 m/s²。
- 组合扰动：PD 增益、质心和物体质量随机化；评估脚本还始终开启摩擦随机化。常规条件关闭前三项，但**仍开启摩擦随机化**，因此不是完全无随机化的测试。
- “完成整圈并保持”定义：已结束的回合累计净旋转不小于 2π，且因达到回合时限而结束；比例以**已结束回合**为分母。高度越界终止单独统计。详见归档的 `source/rl_isaaclab/scripts/evaluate.py`。

| 条件 / 种子 | 模型 | 完成整圈并保持 | 平均净旋转/回合 | 高度越界终止 | 平均回合奖励 |
| --- | --- | ---: | ---: | ---: | ---: |
| 组合扰动 / 42 | 原模型 | 121/138 = 87.68% | 3.09 圈 | 12 | 355.46 |
| 组合扰动 / 42 | 本次 `best.pth` | **124/134 = 92.54%** | 3.39 圈 | **8** | 425.63 |
| 组合扰动 / 42 | 本次 `last.pth` | 122/135 = 90.37% | 3.41 圈 | 8 | 421.80 |
| 组合扰动 / 123 | 原模型 | 116/146 = 79.45% | 2.74 圈 | 24 | 313.76 |
| 组合扰动 / 123 | 本次 `best.pth` | **126/139 = 90.65%** | **3.22 圈** | **12** | 405.83 |
| 组合扰动 / 123 | 本次 `last.pth` | 120/144 = 83.33% | 3.09 圈 | 18 | 383.03 |
| 常规条件 / 42 | 原模型 | 127/134 = 94.78% | 4.05 圈 | 6 | 429.18 |
| 常规条件 / 42 | 本次 `best.pth` | **127/131 = 96.95%** | 4.02 圈 | **3** | 481.36 |

相对原模型，本次 `best.pth` 的组合扰动成功率在两个种子上分别提升 **4.86** 和 **11.20 个百分点**；平均净旋转分别增加约 **0.30** 和 **0.48 圈**。常规条件成功率也略升，但平均净旋转从 4.05 降至 4.02 圈，不能说所有指标都提高。以上均为有限种子的仿真结果，尚未验证真实机械手表现。

原始评估数据：`artifacts/rotation_robust_2026-10-06/run/summary.json`，以及同目录的三个 `*_comparison_seed_*.json` 文件。

## 模型校验与使用

| 归档文件 | SHA-256 |
| --- | --- |
| `run/stage1_nn/best.pth` | `093fd73b0e4d8435efe8e84fb461f63eb3bb289c5a2a8f78a667e91ee32dbb9b` |
| `run/stage1_nn/last.pth` | `5a6a6eef73b22a4c2538f498201c6e34ca8eb41e5adc4a0edc89e05fe3b3950c` |
| `teacher_best.pth` | `994473930f14909aea55eb204c3b21135b9249547797e36c2fc38df357bb290c` |

在项目根目录和已激活的 `sharpa-rl` 环境中可视化推荐模型：

```bash
mkdir -p outputs
python rl_isaaclab/scripts/play.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --num_envs 4 --device cuda:0 --show_axes \
  --load_path artifacts/rotation_robust_2026-10-06/run/stage1_nn/best.pth
```

当前 `play.py` 会删除项目根目录的 `outputs/`；若该目录后来存有需要保留的文件，请先备份再运行。

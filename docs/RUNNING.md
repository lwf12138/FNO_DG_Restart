# 运行与复现

以下命令均从仓库根目录执行。Python 路径和输出目录在代码中相对项目解析，不依赖作者电脑或相邻工程。

## 环境

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

主要依赖已固定版本。`requirements-lock.txt` 是发布核验环境的完整 Python 依赖快照；如需匹配该环境，可改为 `python -m pip install -r requirements-lock.txt`。它不是所有操作系统上的兼容性承诺。

## 冻结模型示例

```bash
python examples/demo.py --case G1_00
python examples/demo.py --case G3_01
```

输出位于 `outputs/demo_G1_00/` 等目录；每次调用真正执行 FNO，而非读取保存预测冒充推理。DG 参考来自已保存的验收结果。三场真值和预测共享纵轴，每场另画绝对误差。重复运行会更新同名示例输出。

`python scripts/verify_release.py` 检查 64 条已公开案例的预测回归、数据与预处理一致性和验收统计。这是发布回归检查，不是新的泛化实验。仅校验文件时使用 `--integrity-only`。

## 从头训练与评估

发布包含 `data/dataset.npz` 和 `fno_training/prepared/`，无需重新生成。检查梯度和训练：

```bash
python fno_training/check_setup.py --device cpu --backward
python fno_training/train_fno.py
```

默认 CPU、2500 epoch；会创建新的 `fno_training/results/train_.../`，不覆盖选定历史实验。快速检查可显式加 `--epochs 2`，但其结果不能代表正式训练。

评估附带的 epoch 2478 基线：

```bash
python fno_training/evaluate_fno.py --split test
```

如评估新训练模型，传入 `--checkpoint results/train_实际运行目录/best.pt`。该路径相对 `fno_training/config.json` 所在目录解析。新评估单独保存；已公开测试数据不得再被描述成未参与开发的隐藏测试集。

默认配置指向历史基线；验收与示例明确使用 A 组冻结模型，两者不要混为同一个 checkpoint。模型接口和物理限制见 [模型说明](MODEL_CARD.md)。

## A/B 损失对照

```bash
python fno_training/controlled_finetune/check_objectives.py
python fno_training/controlled_finetune/finetune_fno_pair.py --config config_A_mse.json
python fno_training/controlled_finetune/finetune_fno_pair.py --config config_B_mixed.json
```

各组均从附带基线加载参数，重置优化器与调度器，分别做 4800 次更新。只按同一验证指标比较，不能比较两个不同训练损失的原始数值。配置中的文件散列已更新到发布副本；历史实验记录保留原散列，并由 `provenance/checkpoint_repackaging.json` 映射。

比较新结果时：

```bash
python fno_training/controlled_finetune/compare_pair.py \
  --a fno_training/controlled_finetune/runs/finetune_A_实际目录 \
  --b fno_training/controlled_finetune/runs/finetune_B_实际目录
```

## DG 编译与数据重算

需要 CMake、C++17 编译器和 Boost 头文件。已有工具则无需重装。macOS 可使用 Homebrew 的 `cmake`、`boost`；Linux 可使用发行版提供的对应开发包。原始数据求解器在检测到本机 libomp 时可选启用 OpenMP；独立验收求解器固定串行。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python scripts/check_dg_release.py
```

最后一条命令编译独立验收求解器，检查两个三模态接口案例，并重新求解一个已保存脉冲案例。它会更新 `fno_acceptance_v1/artifacts/release_smoke/` 下的输出，不替代完整时空收敛研究。

数据全部重算应在另一个工作副本中运行，以保留附带数据和其散列：

```bash
bash run.sh --stage all --workers 2
```

`run.sh` 优先使用环境变量 `DG_PYTHON`，其次是本项目 `.venv/bin/python`，再其次是 PATH 中的 `python3`。它会创建必要目录，不依赖其他项目的虚拟环境。仅预检可用 `--stage preflight`。重生成的压缩容器文件不保证字节级一致；重新训练前应在新实验中重新准备数据并更新协议，不能把新标签套入旧冻结散列。

## 重跑四组验收

```bash
python fno_acceptance_v1/run_acceptance.py --stage all
```

该命令重新准备连续初值，编译隔离 DG，完成参考计算、预测、计时及报告；可能需要较长时间。新输出在 `fno_acceptance_v1/artifacts/`，不会改写 `evidence/acceptance/` 的历史结果。也可依次使用 `prepare`、`reference`、`evaluate`、`benchmark`、`report` 阶段。

历史计时与新计时应分别引用。发布包不包含大量 DG 原始运行目录，历史摘要中的 `<PROJECT_ROOT>/...` 是去标识后的出处记录，不能当作现存文件路径。要核验完整逐步运行过程，应执行新的验收。

## 发布到 GitHub

本目录已包含 `.gitignore`、数据、权重和发布证据。先查看 `git status --short` 与 `docs/PUBLISHING.md`，确认许可证和仓库名称，再提交并推送。不要把原工作目录整体上传，也不要将生成的 `.venv/`、`build/`、`outputs/` 或重复训练目录加入 Git。

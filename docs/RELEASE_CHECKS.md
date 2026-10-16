# 发布核验记录

核验日期：2026-10-16。使用新建的独立 Python 环境，从发布目录安装 requirements.txt；环境为 macOS 15.7.4 / arm64、Python 3.12.14、CPU。完整依赖版本见 requirements-lock.txt。本次没有重新训练或调优。

| 项目 | 结果 |
|---|---|
| 依赖安装及 pip check | 通过，无依赖冲突 |
| 发布文件与冻结来源校验 | 通过；文件清单见 provenance/release_manifest.json |
| 数据、划分、归一化 | 通过，384／64／64；归一化往返最大绝对差约 5.10e-8 |
| FNO 结构、65 点输出、反向传播 | 通过；50,275 个参数、实际保留 9 个 rFFT 系数；未执行优化器更新 |
| A 组加载回归 | 固定 4 条验证样本通过 |
| 四组验收预测回归 | 64 条全部通过；与保存预测的最大绝对差为 0 |
| 基线 test 重评估 | 64 条通过，预测与保存结果完全一致 |
| A/B 配置与目标函数 | 两组配置、损失公式、解析梯度及零范数处理通过；历史比较脚本可重新生成报告 |
| 示例运行 | G1_00 与 G3_01 均生成图、数组、指标；图像已检查 |
| C++ 主程序 | CMake Release 编译通过；AppleClang 17、Boost 1.90.0 |
| DG 数据预检 | run.sh 的 5 个材料预检通过，包含指定 3 个案例的细网格比较 |
| 独立 DG 回归 | 两个三模态接口检查通过；G3_01 参考重算最大绝对差约 1.41e-14 |
| 发布清理 | 已扫描文本及 NPZ 字符串中的个人绝对路径与常见凭证模式；权重环境信息已去路径 |
| 原始工程保留 | 所选 181 个来源文件的原始 SHA256 复核一致 |

机器可读记录见 [release_checks.json](../provenance/release_checks.json)。首次打包文件的字节身份由 manifest 验证；修改文件后完整性检查失败属于预期，不表示科学结果自动失效。开发新实验应保留原发布快照并维护新的版本和记录。

## 可重复的主要检查

从仓库根目录、已激活环境中执行：

```bash
python -m pip check
python scripts/verify_release.py
python fno_training/check_setup.py --device cpu --backward
python fno_training/controlled_finetune/check_objectives.py
python fno_training/evaluate_fno.py --split test
python examples/demo.py --case G1_00
python examples/demo.py --case G3_01
python scripts/check_dg_release.py
DG_PYTHON="$(command -v python)" bash run.sh --stage preflight --workers 2
```

编译器会提示旧 CMake FindBoost 策略警告；本机编译成功，未为消除提示而改动冻结数值源码。

这些检查说明发布副本在上述环境可运行且保留已存结果。没有重跑全部训练、64 条案例的所有 DG 细化与完整计时；未验证 Linux、Windows、CUDA、MPS，也未给出跨环境逐位一致性承诺。科学验收状态仍为 G1/G2 初步通过、人工复核未关闭，G3/G4 未通过。

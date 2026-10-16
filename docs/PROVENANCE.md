# 来源与发布整理

此目录是 FNO_DG_Restart 的独立发布副本，整理于 2026-10-16。原实验目录没有被修改。来源清单的所有路径均相对原项目根目录；不包含个人绝对路径。

- `provenance/source_inventory.json`：被选入文件的原路径、发布路径与整理前 SHA256。
- `provenance/checkpoint_repackaging.json`：checkpoint 新旧散列及参数一致性检查。
- `provenance/release_manifest.json`：当前发布内容的散列，供 `scripts/verify_release.py` 校验。
- `provenance/original_acceptance_config.json`：历史验收配置的去路径副本；运行时使用 `fno_acceptance_v1/config.json`。
- `provenance/original_dataset_README.md`：最初数据交付时的 README，仅作历史记录，其中“未训练”不描述当前项目。

## 保留与省略

保留全部 512 条数据、预处理数组、选定基线及 A/B 的 best/last 权重和训练记录、两份基线评估、64 条冻结验收的输入／预测／参考／指标／图，以及数值源代码。

省略虚拟环境、编译产物、缓存、大量重复续跑、所有逐案例 DG 原始目录和私人任务提示。历史参考与计时摘要仍保留，但其原始日志路径可能指向未随包分发的内容。新运行可重建这些记录，不能将“有摘要”写成“全部原始运行日志随包提供”。

## 改动性质

模型参数、数据数组、归一化和物理协议没有为发布而调优或重训。清理了文本及 checkpoint 环境信息中的个人绝对路径，重新序列化了 checkpoint，因此容器 SHA256 变化。历史指标中引用的旧 SHA256 仍代表原实验身份，不应静默替换成新实验结果。

发布验收配置及其固定 checkpoint 校验常量使用新容器 SHA256，23 项 source_locks 重算为发布副本。A/B 运行配置更新了基线容器散列；历史 A/B experiment 记录保留旧配置身份。`run.sh` 改为本项目环境路径，并补齐目录创建。数学内核和训练模型定义未改变。

历史报告中的个人路径被写为 `<PROJECT_ROOT>` 或 `<USER_HOME>`，这些只是出处占位符，不能用于执行。旧 `hashes.json` 等属于历史记录；发布完整性以 `release_manifest.json` 为准。

`.gitattributes` 禁用自动换行转换，保留数值源码和历史 CSV 的原始字节，避免 Git 检出改变冻结散列。历史文档的相对链接已适配发布目录。

## 数值内核来源

`vendor/base_double/` 是用户原项目保留的双精度 DG 内核快照；`prepare_solver.py` 添加无源三模态初值接口。`fno_acceptance_v1/reference_solver/original/` 保留验收扩展前的内核；新驱动允许完整初始 DG 系数，以连续函数求积初始化一般初值。

RadauIIA-3 在本实现中表示三阶段、名义五阶；不要套用后来另一份 ROADMAP 中两阶段或不同物理范围的设置。原型的参数与数据以当前配置及保存记录为准。

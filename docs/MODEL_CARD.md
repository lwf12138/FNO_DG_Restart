# 模型说明

模型用途为一维 Maxwell–Cole–Cole 的固定终点 T=1 查询。输入为原 65 点完整均匀网格上的 E0/H0 与标量 alpha/tau；P0=0，PEC，零初始历史。输出为同一网格上的物理 E/H/P，形状 `[3,65]`。

示例使用受控 A 组冻结权重；基线和 B 组供实验对照。当前发布 checkpoint 只重写了含个人路径的环境元数据；新旧 SHA256 与参数一致性记录见 [重打包清单](../provenance/checkpoint_repackaging.json)。归一化、模型结构、参数和物理协议均保留。

训练数据为前三模态正弦／余弦初值。新脉冲函数族验收失败。范围内材料数值合法，不代表任意组合或输入函数已经可靠；不支持更换 L、T、边界、物理系数或网格后直接使用。

库版本 neuraloperator 2.0.0、torch 2.8.0，CPU 为发布验证路径。MPS/CUDA 不作此版本的验证声明。不存在未知历史输入、物理残差认证或自动不确定性界。

独立接口可调用 `fno_training.predict_fno.predict_fields(checkpoint_path, alpha, tau, E0, H0, x, device="cpu")`，checkpoint 传绝对路径最明确。完整演示见 [examples/demo.py](../examples/demo.py)。

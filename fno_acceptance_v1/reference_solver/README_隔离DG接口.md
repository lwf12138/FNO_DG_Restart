> 历史技术记录。当前状态和可执行命令以根目录 README 与 docs/RUNNING.md 为准；文中“本轮未执行”等措辞指原交付时点。

# 隔离 DG 接口（已编写，待用户运行）

本目录快照来自现有工程的 `src/cole_cole.cpp`、`include/cole_cole.hpp` 和 `src/dataset_driver.cpp`，原件散列保存在 `source_manifest.json`。未运行原 `prepare_solver.py`，未修改原求解器、vendor、build、缓存或结果。

`original/` 保留逐字节相同的旧源码；CMake 的 `dg_original_regression` 只供旧接口回归。`dg_acceptance` 使用独立的新驱动、带最小初值分支的内核副本。两个 target 都走内核现有串行路径，`-O3`，不启用 OpenMP。空间通量、空间装配（原24点Gauss）、RadauIIA-3、CQ权重与完整 block-FFT 历史、零源项和 P0/Caputo 历史不变。

## 唯一数值接口改动

最小差异见 `cole_cole.hpp.patch` 和 `cole_cole.cpp.patch`：

1. `Config` 增加完整初始系数及显式开关。
2. 仅在 `initial_modal` 原有 sourceFree 分支中，按完整系数直接填入 E/H/P 三个矩阵，然后返回。检查数量、有限值和 P0=0。原三模态投影分支保留，但新驱动使用完整系数入口。
3. 新驱动只请求 T=1 的终点，不调用 `compute_errors` 或制造解误差。内核仍创建其旧 `Exact` 对象，但 sourceFree 时间推进的外源向量保持零，初值也不取自该对象。

Python 中 `initial_conditions.evaluate_initial(spec,x,L)` 是唯一连续初值定义。FNO 接收它在65点的采样；DG 在每个单元的任意Gauss坐标调用同一个函数。不是对65点粗插值，也不是先截断至三个模态。

## 基函数、质量及布局

内核的基是首一 Legendre：φ0=1，φ1=ξ，φ(n+1)=ξφn−n²/(4n²−1)φ(n−1)。令 c_m=φ_m(1)，`mm_m=c_m²/(2m+1)`，物理质量为 `h_j*mm_m`。Python 计算：

`U[j,m] = integral_cell(f(x)*phi_m((x-xc)/J) dx)/(h_j*mm_m)`。

这与旧 `0.5*sum(w*f*phi)/mm` 完全同一投影公式；J=h/2 的映射因子已在物理积分中处理。二次基依次是 1、ξ、ξ²−1/3；不是标准 Legendre P2。

文件布局为 little-endian float64，C-order `[field(E,H,P), cell, mode]`，各场的扁平索引是 `cell*(k+1)+mode`。C++ 回写接收的初始系数，Python逐项精确比较传输前后值。终点65点在内部界面取左右迹算术平均，在物理端点取内侧迹，不硬改E或H。

初始投影每单元分2段，并额外在所有脉冲支撑边缘切开；32点和64点Gauss独立计算。采用64点系数，并以模态质量范数检验32/64点差异。未通过即保存检查并报错，参考状态不能当作可靠。此求积核验是投影检查，不是时间推进的误差保证。

## 用户手动验收

主入口显式启动后，`DGAdapter.build()` 才会调用隔离 CMake；导入、构造对象不会编译。若 CMake/C++17/Boost 缺失会明确报错，请自行安装；不会自动安装依赖。也可手动编译：

```bash
cmake -S reference_solver -B reference_solver/build -DCMAKE_BUILD_TYPE=Release
cmake --build reference_solver/build --parallel 1
```

以上独立命令构建后二进制仍须通过主入口的 `build()` 记录源码/二进制散列，才能用于验收。推荐直接使用主入口的显式构建设置。

主入口调用 `run_regression(manifest['regression_sample_ids'])`，至少两条旧样本在 N64/k2/dt0.0005 上分别运行旧接口和新接口，核对初始DG系数、连续初值65点和终点采样。旧回归额外保存 t=0，**其耗时不用于单终点速度比较**。每次回归写新 UUID 目录，失败记录保留。每个新函数投影另外检查65点与任意形状坐标求值一致，并保存节点/求积值、函数源码和系数散列。主入口再将65点与冻结manifest输入比较。

全部主样本的 N64/dt0.0005 与 N128/dt0.00025 差异，以及预定8条的单独空间、时间细化，由主入口安排。它们是数值自一致性检查，不是解析精确解或严格误差上界。

## 计时与缓存边界

`solve_seconds` 只计 C++ 求解，包括空间组装、CQ权重加载/生成、LU重建、初值系数装入和全部时间步。另记 C++ 初系数读取、终点采样、结果文件写出（不含metadata/stdout）时间。

Python `process_wall_seconds` 包含子进程启动/运行/退出及其输出；`wrapper_wall_seconds` 从连续初值投影开始，到输出读取、数值检查和系数传输核对结束，包含双阶初始求积检查及文件I/O。构建、结果指纹预检、CQ缓存散列和报告写出在此时间之外。连续求值及投影内部的轻量checksum属于投影检查开销。该wrapper是当前可审计的封装口径，不能宣称进程内最优DG端到端耗时。

磁盘只缓存 CQ 权重，键由 alpha、Nt、RK/CQ设置确定；tau 不在无量纲权重键中，材料改变也可能合法命中。每个C++进程都会重新分解系统，不能把此实现说成已有LU常驻复用。缓存文件另以SHA256登记；不接管或清空原工程缓存。基准冷缓存用全新隔离目录，热缓存用相同设置预先一次真实求解填充，然后再次真实求解。两种口径实际 `cache_hit` 都需核对。

`force=False` 可复用散列核对后的终点结果，但返回 `reused_result=True`、wall/projection时间为null，不能用于测速。`force=True` 始终在指定路径下创建 `run_UUID`，不读取旧终点答案。主入口的DG计时必须使用后者。

本轮仅写文件和静态核对；编译、接口回归、初值投影数值检查、求解、细化、设备计时均为**待用户在PyCharm运行**。

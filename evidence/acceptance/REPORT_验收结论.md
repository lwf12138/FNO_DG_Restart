# 冻结 FNO：T=1 四组初步验收

|组别与范围|实际材料/计划样本|E/H/P 平均相对误差|三场 ≤1% / 2% / 5% 观察数（可判数）|R P95 / 最大|弱场绝对误差最大 E/H/P|参考检查|时间与倍率口径|初步判定|
|---|---|---|---|---|---|---|---|---|
|G1：训练见过的材料；原三模态分布中新抽初值|4 / 16|0.823% / 1.16% / 1%|6/16（6） / 12/16（12） / 16/16（16）|3.3% / 3.91%|无弱场 / 无弱场 / 无弱场|SAMPLED_PASS|FNO 0.00074254 s；DG 0.37065 s；S=499；基线 cache_hit wrapper/FNO 常驻 batch=1|数值门槛初步满足；人工复核待完成|
|G2：新的范围内材料；与 G1 配对的相同初值|4 / 16|0.527% / 0.889% / 0.587%|10/16（10） / 14/16（14） / 16/16（16）|2.71% / 4.31%|无弱场 / 无弱场 / 无弱场|SAMPLED_PASS|FNO 0.00074269 s；DG 0.37022 s；S=498；基线 cache_hit wrapper/FNO 常驻 batch=1|数值门槛初步满足；人工复核待完成|
|G3：与 G1 相同材料；新的紧支撑平滑初值|4 / 16|49.2% / 67.3% / 42.1%|0/16（0） / 0/16（0） / 0/16（0）|133% / 145%|无弱场 / 无弱场 / 无弱场|SAMPLED_PASS|FNO 0.00074283 s；DG 0.3726 s；S=502；基线 cache_hit wrapper/FNO 常驻 batch=1|未满足本轮数值门槛|
|G4：与 G2 相同新材料；与 G3 配对的相同初值|4 / 16|49.5% / 66.5% / 41.8%|0/16（0） / 0/16（0） / 0/16（0）|140% / 143%|无弱场 / 无弱场 / 无弱场|SAMPLED_PASS|FNO 0.00074265 s；DG 0.37232 s；S=501；基线 cache_hit wrapper/FNO 常驻 batch=1|未满足本轮数值门槛|

以上观察数分母始终为每组全部计划的 16 条，括号内是其中输入合法、接口通过、参考抽样检查通过且三场相对误差均定义的可判通过数。参考不确定时，即使观察误差较小也不算验收通过；也不能把可判数为零理解为 FNO 误差全都超标。描述性均值保留有限数据，不能覆盖参考状态。

S=DG/FNO；S>1 表示相对所列基线更快，S<1 表示该口径不占优。主表 DG 包含当前 Python 封装/进程/文件 IO；这不是双方相同进程内算法耗时的证明。

初判为本轮工作阈值，不是行业标准。每场平均相对误差 ≤2%，且至少 15/16 条三场同时 ≤5%，并要求无未解决缺失、参考或接口问题。弱场和最差样本须人工复核。

## 结论边界

- 原三模态初值族表现较好，但新函数族尚未满足门槛；不能以延长旧分布训练替代新函数覆盖。
- 每组默认 4 个材料、每材料 4 条初值；配对样本不是独立材料重复。此处计数不是未来总体成功率。
- 新材料仅取本轮 alpha∈[0.35,0.65]、tau∈[0.65,1.6] 的训练凸包内设计；不扩张到整个原参数矩形或边缘。实际坐标保留在 CSV 和 manifest。
- 新函数为本轮宽而平滑的紧支撑脉冲。H 的零均值是控制变量选择；其他 H 均值/常数模态、高频、尖峰和不相容边界不在已验范围。
- 能表示函数空间中的算子、能输入新波形、在训练外波形上实测准确，是三个不同命题。前两个不能替代第三个。
- 仅有 T=1 输出；没有验证全时段 Caputo 残差、逐时刻稳定性、完整能量耗散或 DG 通量误差。DG 细化是数值自一致性检查，不是精确解或严格误差上界。

## 分组明细与待复核项

### G1

状态 `PRELIMINARY_PASS`；数值门槛 `PASS`；实际材料数 4；实际 alpha 范围 [0.3, 0.7]，tau 范围 [0.5, 2.0]。
参考预定检查样本：G1_00, G1_15。
R 最大样本：G1_06；缩放绝对误差最大样本：G1_09。

|场|mean|median|P95|max|有定义数/计划数|弱场数|绝对误差最大|参考范数最小|
|---|---|---|---|---|---|---|---|---|
|E|0.823%|0.601%|1.91%|2.07%|16/16|0|0.0080647|0.20615|
|H|1.16%|0.778%|3.3%|3.91%|16/16|0|0.014843|0.12273|
|P|1%|0.8%|2.18%|2.28%|16/16|0|0.009129|0.13298|

计数：`{"missing_manifest_cases": 0, "nonfinite_prediction": 0, "prediction_record_failure": 0, "nonfinite_reference": 0, "input_unresolved": 0, "reference_uncertain": 0, "reference_incomplete": 0, "relative_undefined": 0, "eligible_for_acceptance": 16, "weak_field_cases": 0}`。
未定义/不确定/缺失 ID：无。
原因：数值门槛之外，弱场/最差样本仍待人工复核。
q_E 中位/P95：2.8393e-16 / 4.2529e-16；q_H 中位/P95：2.6939e-16 / 4.0011e-16。仅据此展示离三模态空间的距离，不认证任意高频。

弱场、未定义误差与最大误差样本（不删除）：

|case|场|reference state|norm_ref|abs_l2|真实 relative|scaled absolute|weak|
|---|---|---|---|---|---|---|---|
|G1_06|E|SAMPLED_PASS|0.83753|0.0029454|0.352%|0.0036455|False|
|G1_06|H|SAMPLED_PASS|0.20116|0.0078717|3.91%|0.0073419|False|
|G1_06|P|SAMPLED_PASS|0.40078|0.0030837|0.769%|0.0064287|False|
|G1_09|E|SAMPLED_PASS|1.382|0.0061566|0.445%|0.00762|False|
|G1_09|H|SAMPLED_PASS|1.1775|0.0095087|0.808%|0.0088688|False|
|G1_09|P|SAMPLED_PASS|1.0044|0.009129|0.909%|0.019032|False|

### G2

状态 `PRELIMINARY_PASS`；数值门槛 `PASS`；实际材料数 4；实际 alpha 范围 [0.3738858592378832, 0.5765593285190812]，tau 范围 [0.7097888781822598, 1.549425307524447]。
参考预定检查样本：G2_00, G2_15。
R 最大样本：G2_14；缩放绝对误差最大样本：G2_09。

|场|mean|median|P95|max|有定义数/计划数|弱场数|绝对误差最大|参考范数最小|
|---|---|---|---|---|---|---|---|---|
|E|0.527%|0.395%|1.14%|1.27%|16/16|0|0.0051395|0.20494|
|H|0.889%|0.568%|2.71%|4.31%|16/16|0|0.0097145|0.12317|
|P|0.587%|0.47%|1.14%|1.2%|16/16|0|0.0037509|0.13385|

计数：`{"missing_manifest_cases": 0, "nonfinite_prediction": 0, "prediction_record_failure": 0, "nonfinite_reference": 0, "input_unresolved": 0, "reference_uncertain": 0, "reference_incomplete": 0, "relative_undefined": 0, "eligible_for_acceptance": 16, "weak_field_cases": 0}`。
未定义/不确定/缺失 ID：无。
原因：数值门槛之外，弱场/最差样本仍待人工复核。
q_E 中位/P95：2.8393e-16 / 4.2529e-16；q_H 中位/P95：2.6939e-16 / 4.0011e-16。仅据此展示离三模态空间的距离，不认证任意高频。

弱场、未定义误差与最大误差样本（不删除）：

|case|场|reference state|norm_ref|abs_l2|真实 relative|scaled absolute|weak|
|---|---|---|---|---|---|---|---|
|G2_09|E|SAMPLED_PASS|1.4374|0.0042232|0.294%|0.005227|False|
|G2_09|H|SAMPLED_PASS|1.162|0.0097145|0.836%|0.0090608|False|
|G2_09|P|SAMPLED_PASS|0.88939|0.0037509|0.422%|0.0078198|False|
|G2_14|E|SAMPLED_PASS|0.41091|0.0028126|0.684%|0.0034812|False|
|G2_14|H|SAMPLED_PASS|0.12317|0.0053138|4.31%|0.0049562|False|
|G2_14|P|SAMPLED_PASS|0.21002|0.0019662|0.936%|0.0040991|False|

### G3

状态 `PRELIMINARY_FAIL`；数值门槛 `FAIL`；实际材料数 4；实际 alpha 范围 [0.3, 0.7]，tau 范围 [0.5, 2.0]。
参考预定检查样本：G3_01, G3_11。
R 最大样本：G3_08；缩放绝对误差最大样本：G3_12。

|场|mean|median|P95|max|有定义数/计划数|弱场数|绝对误差最大|参考范数最小|
|---|---|---|---|---|---|---|---|---|
|E|49.2%|39.9%|98.9%|123%|16/16|0|0.62607|0.40555|
|H|67.3%|56.6%|128%|145%|16/16|0|0.90377|0.44962|
|P|42.1%|29.2%|96.4%|129%|16/16|0|0.20378|0.15741|

计数：`{"missing_manifest_cases": 0, "nonfinite_prediction": 0, "prediction_record_failure": 0, "nonfinite_reference": 0, "input_unresolved": 0, "reference_uncertain": 0, "reference_incomplete": 0, "relative_undefined": 0, "eligible_for_acceptance": 16, "weak_field_cases": 0}`。
未定义/不确定/缺失 ID：无。
原因：At least one field mean exceeds its predeclared relative-error limit; Joint threshold count 0/16; requires 15/16。
q_E 中位/P95：0.45057 / 0.71667；q_H 中位/P95：0.77313 / 0.94048。仅据此展示离三模态空间的距离，不认证任意高频。

弱场、未定义误差与最大误差样本（不删除）：

|case|场|reference state|norm_ref|abs_l2|真实 relative|scaled absolute|weak|
|---|---|---|---|---|---|---|---|
|G3_08|E|SAMPLED_PASS|0.96404|0.30398|31.5%|0.37624|False|
|G3_08|H|SAMPLED_PASS|0.60871|0.88554|145%|0.82595|False|
|G3_08|P|SAMPLED_PASS|0.72109|0.15574|21.6%|0.32468|False|
|G3_12|E|SAMPLED_PASS|0.6752|0.30148|44.7%|0.37315|False|
|G3_12|H|SAMPLED_PASS|1.0092|0.90377|89.6%|0.84295|False|
|G3_12|P|SAMPLED_PASS|0.32137|0.1242|38.6%|0.25892|False|

### G4

状态 `PRELIMINARY_FAIL`；数值门槛 `FAIL`；实际材料数 4；实际 alpha 范围 [0.3738858592378832, 0.5765593285190812]，tau 范围 [0.7097888781822598, 1.549425307524447]。
参考预定检查样本：G4_01, G4_11。
R 最大样本：G4_08；缩放绝对误差最大样本：G4_08。

|场|mean|median|P95|max|有定义数/计划数|弱场数|绝对误差最大|参考范数最小|
|---|---|---|---|---|---|---|---|---|
|E|49.5%|39.4%|99.9%|127%|16/16|0|0.60141|0.40912|
|H|66.5%|56%|125%|143%|16/16|0|0.9094|0.4493|
|P|41.8%|30.7%|90.7%|139%|16/16|0|0.26903|0.1938|

计数：`{"missing_manifest_cases": 0, "nonfinite_prediction": 0, "prediction_record_failure": 0, "nonfinite_reference": 0, "input_unresolved": 0, "reference_uncertain": 0, "reference_incomplete": 0, "relative_undefined": 0, "eligible_for_acceptance": 16, "weak_field_cases": 0}`。
未定义/不确定/缺失 ID：无。
原因：At least one field mean exceeds its predeclared relative-error limit; Joint threshold count 0/16; requires 15/16。
q_E 中位/P95：0.45057 / 0.71667；q_H 中位/P95：0.77313 / 0.94048。仅据此展示离三模态空间的距离，不认证任意高频。

弱场、未定义误差与最大误差样本（不删除）：

|case|场|reference state|norm_ref|abs_l2|真实 relative|scaled absolute|weak|
|---|---|---|---|---|---|---|---|
|G4_08|E|SAMPLED_PASS|0.99902|0.31266|31.3%|0.38699|False|
|G4_08|H|SAMPLED_PASS|0.63805|0.9094|143%|0.8482|False|
|G4_08|P|SAMPLED_PASS|0.63371|0.13601|21.5%|0.28356|False|

## 计时完整口径

计时状态：`COMPLETE`。冷启动：`{"initialization_seconds": 0.6864927920978516, "first_prediction_seconds": 0.014205957995727658, "combined_seconds": 0.7006987500935793, "scope": "checkpoint read + strict original model construction/load + first end-to-end call; dependency imports/hash checks/process startup excluded; OS file cache uncontrolled", "first_query": "first fixed A-validation regression sample; same T=1/65-point task"}`。

FNO 为常驻 batch=1，包含编码、tensor 转换、前向与反归一化。DG solve 与 wrapper 分列；缓存命中状态以实际记录为准，LU/分解是否重建以 timing_protocol 为准。编译、散列、画图和保存结果不计入查询延迟。

只发布 benchmark 已保存且 FNO/对应 DG 计时均完整的倍率。预热不足、计时样本不足、部分求解失败或缓存状态不符时，有限时长仍展示，但不以中位数重新计算倍率。

|组|设置|缓存|口径|median s|P95 s|max s|次数|计时状态|已验证倍率|
|---|---|---|---|---|---|---|---|---|---|
|G1|FNO|resident|batch=1 end-to-end|0.00074254|0.00075686|0.0010194|200|COMPLETE|—|
|G1|baseline|cache_hit|wrapper|0.37065|0.38901|0.39427|6|COMPLETE|499.17|
|G1|baseline|cache_hit|solve|0.35321|0.37067|0.3757|6|COMPLETE|475.68|
|G1|baseline|cache_hit|projection|0.0098107|0.010172|0.01024|6|COMPLETE|不单独称加速|
|G1|baseline|cache_miss|wrapper|0.40832|0.41022|0.41038|6|COMPLETE|549.89|
|G1|baseline|cache_miss|solve|0.39111|0.39301|0.39311|6|COMPLETE|526.72|
|G1|baseline|cache_miss|projection|0.009767|0.009817|0.0098275|6|COMPLETE|不单独称加速|
|G1|reference|cache_hit|wrapper|1.6131|2.0657|2.079|6|COMPLETE|2172.4|
|G1|reference|cache_hit|solve|1.5851|2.0382|2.0523|6|COMPLETE|2134.7|
|G1|reference|cache_hit|projection|0.018183|0.019144|0.019296|6|COMPLETE|不单独称加速|
|G1|reference|cache_miss|wrapper|1.6635|1.6825|1.6882|6|COMPLETE|2240.3|
|G1|reference|cache_miss|solve|1.6363|1.6562|1.6622|6|COMPLETE|2203.6|
|G1|reference|cache_miss|projection|0.018292|0.0184|0.018421|6|COMPLETE|不单独称加速|
|G2|FNO|resident|batch=1 end-to-end|0.00074269|0.00075629|0.00084233|200|COMPLETE|—|
|G2|baseline|cache_hit|wrapper|0.37022|0.37122|0.37125|6|COMPLETE|498.49|
|G2|baseline|cache_hit|solve|0.35304|0.35404|0.35409|6|COMPLETE|475.36|
|G2|baseline|cache_hit|projection|0.0098375|0.010009|0.010046|6|COMPLETE|不单独称加速|
|G2|baseline|cache_miss|wrapper|0.41019|0.41082|0.41085|6|COMPLETE|552.31|
|G2|baseline|cache_miss|solve|0.39302|0.39356|0.39367|6|COMPLETE|529.18|
|G2|baseline|cache_miss|projection|0.0097226|0.0098807|0.0098846|6|COMPLETE|不单独称加速|
|G2|reference|cache_hit|wrapper|1.732|1.9478|1.9836|6|COMPLETE|2332.1|
|G2|reference|cache_hit|solve|1.7038|1.9181|1.9549|6|COMPLETE|2294.1|
|G2|reference|cache_hit|projection|0.018361|0.019782|0.019922|6|COMPLETE|不单独称加速|
|G2|reference|cache_miss|wrapper|1.6656|1.7227|1.74|6|COMPLETE|2242.7|
|G2|reference|cache_miss|solve|1.6396|1.6967|1.7141|6|COMPLETE|2207.6|
|G2|reference|cache_miss|projection|0.018235|0.018301|0.018311|6|COMPLETE|不单独称加速|
|G3|FNO|resident|batch=1 end-to-end|0.00074283|0.00075277|0.00079517|200|COMPLETE|—|
|G3|baseline|cache_hit|wrapper|0.3726|0.39337|0.39918|6|COMPLETE|501.6|
|G3|baseline|cache_hit|solve|0.35394|0.37186|0.37766|6|COMPLETE|476.48|
|G3|baseline|cache_hit|projection|0.013434|0.015386|0.015419|6|COMPLETE|不单独称加速|
|G3|baseline|cache_miss|wrapper|0.41201|0.42273|0.42523|6|COMPLETE|554.65|
|G3|baseline|cache_miss|solve|0.39264|0.40144|0.40407|6|COMPLETE|528.57|
|G3|baseline|cache_miss|projection|0.013432|0.015172|0.015184|6|COMPLETE|不单独称加速|
|G3|reference|cache_hit|wrapper|1.6308|2.0119|2.0142|6|COMPLETE|2195.3|
|G3|reference|cache_hit|solve|1.5984|1.9789|1.9826|6|COMPLETE|2151.8|
|G3|reference|cache_hit|projection|0.025921|0.028991|0.029177|6|COMPLETE|不单独称加速|
|G3|reference|cache_miss|wrapper|1.6727|1.7088|1.7155|6|COMPLETE|2251.8|
|G3|reference|cache_miss|solve|1.6421|1.6717|1.678|6|COMPLETE|2210.6|
|G3|reference|cache_miss|projection|0.024969|0.029923|0.030463|6|COMPLETE|不单独称加速|
|G4|FNO|resident|batch=1 end-to-end|0.00074265|0.00075459|0.00078588|200|COMPLETE|—|
|G4|baseline|cache_hit|wrapper|0.37232|0.37564|0.37592|6|COMPLETE|501.35|
|G4|baseline|cache_hit|solve|0.35298|0.35436|0.35459|6|COMPLETE|475.3|
|G4|baseline|cache_hit|projection|0.013414|0.015231|0.015253|6|COMPLETE|不单独称加速|
|G4|baseline|cache_miss|wrapper|0.41152|0.41325|0.41326|6|COMPLETE|554.13|
|G4|baseline|cache_miss|solve|0.39208|0.39256|0.39264|6|COMPLETE|527.95|
|G4|baseline|cache_miss|projection|0.013552|0.01513|0.015156|6|COMPLETE|不单独称加速|
|G4|reference|cache_hit|wrapper|1.5979|2.1458|2.1846|6|COMPLETE|2151.6|
|G4|reference|cache_hit|solve|1.5575|2.1102|2.1485|6|COMPLETE|2097.2|
|G4|reference|cache_hit|projection|0.025276|0.036901|0.039817|6|COMPLETE|不单独称加速|
|G4|reference|cache_miss|wrapper|1.6895|1.7206|1.7224|6|COMPLETE|2274.9|
|G4|reference|cache_miss|solve|1.6573|1.6844|1.6862|6|COMPLETE|2231.6|
|G4|reference|cache_miss|projection|0.025176|0.030631|0.031445|6|COMPLETE|不单独称加速|

基线/参考设置与计时环境完整记录：

```json
{
  "configured_dg": {
    "build_if_missing": true,
    "threads": 1,
    "projection_order": 32,
    "projection_check_order": 64,
    "projection_subcells": 2,
    "projection_atol": 1e-12,
    "projection_rtol": 1e-10,
    "baseline": {
      "N": 64,
      "k": 2,
      "dt": 0.0005,
      "T": 1.0,
      "points": 65
    },
    "reference": {
      "N": 128,
      "k": 2,
      "dt": 0.00025,
      "T": 1.0,
      "points": 65
    },
    "space_refinement": {
      "N": 256,
      "k": 2,
      "dt": 0.00025,
      "T": 1.0,
      "points": 65
    },
    "time_refinement": {
      "N": 128,
      "k": 2,
      "dt": 0.000125,
      "T": 1.0,
      "points": 65
    },
    "additional_refinements": 0
  },
  "timing_protocol": {
    "fno_batch": 1,
    "fno_warmups": 20,
    "fno_measured_per_group": 200,
    "fno_scope": "resident encoding/input validation + tensor conversion + forward + NumPy transfer + decoding; no answer cache",
    "cold_scope": "checkpoint read + strict original model construction/load + first end-to-end call; dependency imports/hash checks/process startup excluded; OS file cache uncontrolled",
    "threads": 1,
    "device": "cpu",
    "representative_case_ids": [
      "G1_00",
      "G1_15",
      "G2_00",
      "G2_15",
      "G3_01",
      "G3_11",
      "G4_01",
      "G4_11"
    ],
    "dg_repeats_per_case_per_cache_mode": 3,
    "dg_configurations": {
      "baseline": {
        "N": 64,
        "k": 2,
        "dt": 0.0005,
        "T": 1.0,
        "points": 65
      },
      "reference": {
        "N": 128,
        "k": 2,
        "dt": 0.00025,
        "T": 1.0,
        "points": 65
      }
    },
    "dg_wrapper_scope": "current Python wrapper including continuous initial L2 projection/checks/checksums, process start, IO, solve, sampling and result validation; external provenance/CQ cache hash bookkeeping excluded",
    "dg_solve_scope": "C++ solve_seconds only; Python initial projection/process/IO and terminal sampling excluded",
    "dg_cache_scope": "disk CQ weights only; factorization is rebuilt in every process; OS filesystem cache uncontrolled",
    "fixed_material_query": "warm cache primed with another initial condition at identical alpha/tau; answers never reused",
    "new_material_query": "fresh isolated CQ directory for every cache-miss query",
    "speedup_scope": "median DG wrapper (or separate solve-only diagnostic) / median resident FNO, same two representatives per group",
    "restriction": "current wrappers and two DG configurations; not proof against optimal same-accuracy DG; timing small representative set only",
    "fno_successful_warmups": 20,
    "fno_warmup_status": "COMPLETE",
    "build_info": {
      "commands": [
        [
          "cmake",
          "-S",
          "<PROJECT_ROOT>/fno_acceptance_v1/reference_solver",
          "-B",
          "<PROJECT_ROOT>/fno_acceptance_v1/reference_solver/build",
          "-DCMAKE_BUILD_TYPE=Release"
        ],
        [
          "cmake",
          "--build",
          "<PROJECT_ROOT>/fno_acceptance_v1/reference_solver/build",
          "--parallel",
          "1"
        ]
      ],
      "source_sha256": {
        "CMakeLists.txt": "f30b905390fbdd225910f9c5ebe4bdcc1894d27d1a6bad53714927d7fad56b6d",
        "include/cole_cole.hpp": "e8f0b0858851d23685f3d912b126dc4b2b9171bd3644c56f36fd5448f3927020",
        "original/CMakeLists.txt": "a94bb609951ba76d69e01ffe9ffde5a48b24d46b65917021d4f85f0f565576dc",
        "original/include/cole_cole.hpp": "848b8993027575e9b34ea9cfcbdee8511c78e56d90a93e7b0a35ff769f2569ff",
        "original/src/cole_cole.cpp": "12d000f4a2edbb5ee5fb6cbc781148c7d78a9d37376b3ab5516915329133b309",
        "original/src/dataset_driver.cpp": "836dc69d3d2ce4830f6f362c0e528b6cbe6f81863b9348ea8fc53d1241d90555",
        "src/acceptance_driver.cpp": "07e0776681662744aa8db56c717c12e1e6c0abb79c1eb9e43c54ebe6727a3c4d",
        "src/cole_cole.cpp": "17c68a1017bf6e8199d1fb521ecb49196d4dc6dc39c8f8cb7fd3f5a65bb08880"
      },
      "binary_sha256": {
        "dg_acceptance": "783f90a79c85957ac3d3dd3aad5423c577a3d60dd4230d9c6935e4f4f96b0013",
        "dg_original_regression": "22e3ebe637f90a41b656b56165b705afe265395371d46d3da867d74529b9c1b0"
      },
      "build_seconds": 6.042349207913503,
      "reused_build": false,
      "log": "<PROJECT_ROOT>/fno_acceptance_v1/reference_solver/build/build_ccee753eb3084360aac6a9240609b855.log",
      "threads": 1
    }
  }
}
```

不宣称同精度最优 DG 也比 FNO 慢。基线与参考的逐样本差异、空间与时间分开细化结果均保留在 per_case_metrics.csv/reference_checks.json。

## 端点与积分诊断

E 仅比较网络端点与 DG 内侧迹，未强制修改预测。H 积分分别保留初始连续积分、65 点梯形积分、终点预测/参考梯形积分；其差异包含节点求积误差，不能直接解释为连续守恒定理失效。

|case|E左端差|E右端差|H初始连续积分|H初始节点积分|H预测末积分|H参考末积分|
|---|---|---|---|---|---|---|
|G1_00|0.002754|0.0019173|-1.1102e-16|-1.1102e-16|0.0018835|-3.0862e-13|
|G1_01|0.0030993|-0.00083351|1.1102e-16|1.4225e-16|0.0068997|-3.4994e-13|
|G1_02|0.00029312|1.096e-05|0|1.6653e-16|-0.0044957|7.2158e-14|
|G1_03|-0.00070554|0.0069728|2.2204e-16|9.0206e-17|0.0036106|-6.746e-14|
|G1_04|0.0013368|0.0022846|-1.1102e-16|-1.7694e-16|0.0026737|-2.3359e-13|
|G1_05|-0.0038638|-0.0063523|2.2204e-16|-2.6368e-16|0.0060863|3.9206e-13|
|G1_06|0.0024289|1.6847e-05|-1.1102e-16|-1.2837e-16|0.0079542|9.1947e-14|
|G1_07|-0.0013035|-0.0035553|-1.1102e-16|-1.1102e-16|-0.0035267|2.9338e-13|
|G1_08|0.007537|0.00051406|0|2.5674e-16|-0.028949|-1.078e-12|
|G1_09|0.011262|-0.0019848|-4.4409e-16|-2.3592e-16|0.015018|7.6622e-13|
|G1_10|0.0056788|-0.0032082|-2.2204e-16|-8.3267e-17|-0.0023566|6.2948e-13|
|G1_11|-0.0016085|-0.00068481|0|3.1225e-17|0.0058749|4.9754e-13|
|G1_12|-0.0013516|0.0024183|2.2204e-16|4.1633e-17|0.0011293|4.2891e-13|
|G1_13|-0.0023583|-0.00021347|1.1102e-16|7.2858e-17|-0.002112|2.447e-13|
|G1_14|-0.00015342|0.0025775|2.7756e-17|6.5919e-17|0.0039001|-1.8468e-13|
|G1_15|0.0016615|0.0025067|8.3267e-17|7.2858e-17|0.0039319|-4.7085e-13|
|G2_00|0.00060995|0.0022862|-1.1102e-16|-1.1102e-16|-0.0012821|1.3239e-12|
|G2_01|-0.0016208|-0.0014121|1.1102e-16|1.4225e-16|-0.00091003|-4.3941e-13|
|G2_02|0.0010233|-0.00047467|0|1.6653e-16|0.0036052|-1.255e-12|
|G2_03|-0.00090442|0.0041832|2.2204e-16|9.0206e-17|0.0026823|-2.0859e-12|
|G2_04|2.8675e-05|0.0016223|-1.1102e-16|-1.7694e-16|-0.00063582|-2.9901e-13|
|G2_05|0.001257|0.0011253|2.2204e-16|-2.6368e-16|0.0051823|-1.0815e-12|
|G2_06|0.001317|0.0012533|-1.1102e-16|-1.2837e-16|-0.0025066|5.2606e-14|
|G2_07|-0.00046103|0.00015058|-1.1102e-16|-1.1102e-16|-0.008472|-6.5643e-13|
|G2_08|0.0037229|-0.00049961|0|2.5674e-16|-0.0052409|6.4818e-13|
|G2_09|0.0078313|0.0018019|-4.4409e-16|-2.3592e-16|0.016455|-6.4079e-13|
|G2_10|0.0032909|0.00097923|-2.2204e-16|-8.3267e-17|0.0030814|-5.3416e-13|
|G2_11|-0.0040605|-0.00019051|0|3.1225e-17|0.0016631|-4.409e-13|
|G2_12|-0.0011681|0.00060024|2.2204e-16|4.1633e-17|-0.0010864|-2.5594e-14|
|G2_13|-0.002502|-0.0014885|1.1102e-16|7.2858e-17|0.0023844|-2.6826e-13|
|G2_14|0.0008017|0.0022599|2.7756e-17|6.5919e-17|0.0099086|-5.289e-14|
|G2_15|0.0021053|0.00017172|8.3267e-17|7.2858e-17|-0.0031575|-4.7533e-13|
|G3_00|0.063632|-0.0091237|-1.5654e-14|0.0017334|0.017932|0.00013984|
|G3_01|0.047241|-0.052142|-1.1102e-16|-0.00072633|0.021475|-7.3083e-05|
|G3_02|0.068667|-0.064237|-1.1102e-16|-0.0013784|0.0082202|-0.00012854|
|G3_03|-0.011711|-0.020249|-5.5511e-17|0.00083477|-0.079708|7.1922e-05|
|G3_04|-0.029098|-0.11817|2.7756e-17|0.00069854|-0.23057|8.855e-06|
|G3_05|0.042388|0.0068715|0|0.00043469|-0.025669|3.378e-05|
|G3_06|-0.035263|0.049797|-1.249e-16|-0.00072722|-0.1052|-7.955e-06|
|G3_07|-0.040911|-0.0068178|0|0.00075426|0.034443|7.7046e-06|
|G3_08|-0.042133|-0.027725|5.5511e-16|0.00039374|0.1325|1.106e-05|
|G3_09|0.020493|0.052668|1.1102e-16|-2.4399e-05|0.036539|-9.1509e-06|
|G3_10|-0.038817|0.0063069|1.1796e-15|-0.0011592|-0.049832|-5.4471e-05|
|G3_11|0.01834|0.0053185|1.1102e-16|-0.0027285|0.042032|-0.00012389|
|G3_12|-0.028065|-0.060068|2.2204e-16|0.0011447|0.057903|0.00018752|
|G3_13|-0.031061|0.033312|0|-0.00081843|0.11514|-0.00014189|
|G3_14|-0.0054105|-0.052411|-5.5511e-17|-0.00096892|-0.068436|-0.00016744|
|G3_15|0.0078176|-0.013367|-2.9421e-15|0.0043754|-0.039348|0.00076675|
|G4_00|0.059874|-0.0083598|-1.5654e-14|0.0017334|0.023273|0.00022651|
|G4_01|0.044394|-0.050348|-1.1102e-16|-0.00072633|0.036555|-0.00010314|
|G4_02|0.064799|-0.06288|-1.1102e-16|-0.0013784|0.016864|-0.00019757|
|G4_03|-0.013272|-0.021372|-5.5511e-17|0.00083477|-0.083527|0.00011735|
|G4_04|-0.026322|-0.10997|2.7756e-17|0.00069854|-0.21475|0.00010617|
|G4_05|0.037645|0.0055831|0|0.00043469|-0.012208|7.2911e-05|
|G4_06|-0.035626|0.043324|-1.249e-16|-0.00072722|-0.1153|-9.8376e-05|
|G4_07|-0.043361|0.0011708|0|0.00075426|0.052317|0.00011334|
|G4_08|-0.042892|-0.033249|5.5511e-16|0.00039374|0.1373|2.6553e-05|
|G4_09|0.018481|0.050891|1.1102e-16|-2.4399e-05|0.034044|-9.2221e-06|
|G4_10|-0.039536|0.012978|1.1796e-15|-0.0011592|-0.030756|-0.00010141|
|G4_11|0.01877|0.0034579|1.1102e-16|-0.0027285|0.079026|-0.00023234|
|G4_12|-0.030348|-0.063151|2.2204e-16|0.0011447|0.044425|0.00010013|
|G4_13|-0.028245|0.035101|0|-0.00081843|0.11869|-7.2699e-05|
|G4_14|-0.0087565|-0.05716|-5.5511e-17|-0.00096892|-0.087427|-7.4282e-05|
|G4_15|0.0082667|-0.011881|-2.9421e-15|0.0043754|-0.039525|0.00041097|

## 图与产物

绘图状态：`COMPLETE`。预固定代表样本、R 最大样本、缩放绝对误差最大样本及未解决样本均保留；每场单独绘图。

全部 64 条计划样本的字段、ID、初值参数、q、状态和数值检查见 per_case_metrics.csv、cases_manifest.json 与 reference_checks.json；predictions_and_reference.npz 保留未修改的物理尺度数组，失败用 NaN。

接口检查：`{"dg_regression": {"status": "PASS", "sample_count": 2, "samples": [{"sample_id": "sample_0000", "status": "PASS", "initial_coefficient_max_abs_difference": [2.7682012498225772e-14, 3.792409095093596e-14, 0.0], "initial_coefficient_tolerance": [5.109317706717163e-11, 7.124753379067007e-11, 1e-12], "endpoint_absolute_l2": [1.0403118038155522e-14, 1.4556997111578905e-14, 4.674178056590102e-15], "endpoint_tolerance": [5.912836130941798e-10, 1.1326753365313293e-09, 3.7155184927873645e-10], "continuous_65_vs_original_initial_max_abs": [1.1102230246251565e-16, 2.220446049250313e-16, 0.0], "new_output_dir": "<PROJECT_ROOT>/fno_acceptance_v1/artifacts/dg/interface_regression/run_9f651e901bc549fa8f8a5b5c8362eab1/sample_0000/new", "old_output_dir": "<PROJECT_ROOT>/fno_acceptance_v1/artifacts/dg/interface_regression/run_9f651e901bc549fa8f8a5b5c8362eab1/sample_0000/old"}, {"sample_id": "sample_0001", "status": "PASS", "initial_coefficient_max_abs_difference": [2.1784928461859865e-14, 9.683118022685289e-15, 0.0], "initial_coefficient_tolerance": [6.45758746793253e-11, 1.868251186992994e-11, 1e-12], "endpoint_absolute_l2": [6.642197876003334e-15, 1.1223643831762841e-14, 2.5639902738109247e-15], "endpoint_tolerance": [7.609895882251826e-10, 2.3158346785221462e-10, 4.625762643147229e-10], "continuous_65_vs_original_initial_max_abs": [1.1102230246251565e-16, 1.6653345369377348e-16, 0.0], "new_output_dir": "<PROJECT_ROOT>/fno_acceptance_v1/artifacts/dg/interface_regression/run_9f651e901bc549fa8f8a5b5c8362eab1/sample_0001/new", "old_output_dir": "<PROJECT_ROOT>/fno_acceptance_v1/artifacts/dg/interface_regression/run_9f651e901bc549fa8f8a5b5c8362eab1/sample_0001/old"}], "dataset_sha256": "a0bc9598eb028ff63c3eceafd5ebefb956591849c654257b2caebc00f3e24f96", "scope": "same DG kernel/config; original sine/cosine projection vs full-coefficient interface", "output_dir": "<PROJECT_ROOT>/fno_acceptance_v1/artifacts/dg/interface_regression/run_9f651e901bc549fa8f8a5b5c8362eab1", "not_a_reference_error_bound": true}, "fno_regression": {"status": "PASS", "samples": [{"sample_id": "sample_0056", "max_absolute_difference": 5.379383366532764e-07, "pass": true}, {"sample_id": "sample_0057", "max_absolute_difference": 5.098905466738302e-07, "pass": true}, {"sample_id": "sample_0058", "max_absolute_difference": 3.0593432798209363e-07, "pass": true}, {"sample_id": "sample_0059", "max_absolute_difference": 3.059343280376048e-07, "pass": true}], "scope": "frozen A loading/encoding/decoding regression only; these are not any of the 64 new acceptance cases", "checkpoint": {"checkpoint_sha256": "f7ddd5f2517487935441aa1dd50e8a465212a7e4ca21ee3e17cf09f155f504f9", "epoch": 176, "fine_tune_step": 4224, "source_epoch": 2478, "model_config": {"in_channels": 5, "out_channels": 3, "hidden_channels": 32, "n_layers": 4, "n_modes": [16], "positional_embedding": null, "factorization": null, "norm": null, "activation": "gelu", "precision": "float32", "domain_padding": 0.125, "use_channel_mlp": true, "channel_mlp_expansion": 0.5, "channel_mlp_dropout": 0.0, "channel_mlp_skip": "soft-gating", "fno_skip": "linear", "lifting_channel_ratio": 2, "projection_channel_ratio": 2, "implementation": "reconstructed", "fno_block_precision": "full"}, "environment": {"python": "3.12.14 (main, Aug 12 2026, 13:57:54) [Clang 17.0.0 (clang-1700.6.4.2)]", "python_executable": "<PROJECT_ROOT>/fno_training/.venv/bin/python", "platform": "macOS-15.7.4-arm64-arm-64bit", "machine": "arm64", "versions": {"torch": "2.8.0", "neuraloperator": "2.0.0", "numpy": "2.0.2", "scipy": "1.13.1", "tensorly": "0.9.0", "tensorly-torch": "0.5.0"}, "selected_device": "cpu", "mps_fallback_environment": "unset"}, "intraop_threads": 1, "interop_threads": 1, "dtype": "float32", "device": "cpu", "answer_cache": false, "cold_scope": "checkpoint read + strict original model construction/load + first end-to-end call; dependency imports/hash checks/process startup excluded; OS file cache uncontrolled"}, "cold_start": {"initialization_seconds": 0.6864927920978516, "first_prediction_seconds": 0.014205957995727658, "combined_seconds": 0.7006987500935793, "scope": "checkpoint read + strict original model construction/load + first end-to-end call; dependency imports/hash checks/process startup excluded; OS file cache uncontrolled", "first_query": "first fixed A-validation regression sample; same T=1/65-point task"}}}`。

![G1_00 E](plots/G1_00_E.png)

![G1_00 H](plots/G1_00_H.png)

![G1_00 P](plots/G1_00_P.png)

![G1_15 E](plots/G1_15_E.png)

![G1_15 H](plots/G1_15_H.png)

![G1_15 P](plots/G1_15_P.png)

![G2_00 E](plots/G2_00_E.png)

![G2_00 H](plots/G2_00_H.png)

![G2_00 P](plots/G2_00_P.png)

![G2_15 E](plots/G2_15_E.png)

![G2_15 H](plots/G2_15_H.png)

![G2_15 P](plots/G2_15_P.png)

![G3_01 E](plots/G3_01_E.png)

![G3_01 H](plots/G3_01_H.png)

![G3_01 P](plots/G3_01_P.png)

![G3_11 E](plots/G3_11_E.png)

![G3_11 H](plots/G3_11_H.png)

![G3_11 P](plots/G3_11_P.png)

![G4_01 E](plots/G4_01_E.png)

![G4_01 H](plots/G4_01_H.png)

![G4_01 P](plots/G4_01_P.png)

![G4_11 E](plots/G4_11_E.png)

![G4_11 H](plots/G4_11_H.png)

![G4_11 P](plots/G4_11_P.png)

![G1_06 E](plots/G1_06_E.png)

![G1_06 H](plots/G1_06_H.png)

![G1_06 P](plots/G1_06_P.png)

![G1_09 E](plots/G1_09_E.png)

![G1_09 H](plots/G1_09_H.png)

![G1_09 P](plots/G1_09_P.png)

![G2_14 E](plots/G2_14_E.png)

![G2_14 H](plots/G2_14_H.png)

![G2_14 P](plots/G2_14_P.png)

![G2_09 E](plots/G2_09_E.png)

![G2_09 H](plots/G2_09_H.png)

![G2_09 P](plots/G2_09_P.png)

![G3_08 E](plots/G3_08_E.png)

![G3_08 H](plots/G3_08_H.png)

![G3_08 P](plots/G3_08_P.png)

![G3_12 E](plots/G3_12_E.png)

![G3_12 H](plots/G3_12_H.png)

![G3_12 P](plots/G3_12_P.png)

![G4_08 E](plots/G4_08_E.png)

![G4_08 H](plots/G4_08_H.png)

![G4_08 P](plots/G4_08_P.png)

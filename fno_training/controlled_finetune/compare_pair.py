"""Compare completed paired runs using saved JSON/CSV only; never import a model.

Run manually after both arms complete:
    python compare_pair.py --a /path/to/A_run --b /path/to/B_run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FIELDS = ("E", "H", "P")
STATS = ("mean", "median", "p95", "max")
RTOL = 1e-7
ATOL = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def number(value: Any, label: str, *, nonnegative: bool = True) -> float:
    require(not isinstance(value, bool), f"{label}: 布尔值不能代替数值。")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: 需要有限数值。") from error
    require(math.isfinite(result), f"{label}: 数值必须有限。")
    if nonnegative:
        require(result >= 0, f"{label}: 数值不能为负。")
    return result


def integer(value: Any, label: str) -> int:
    result = number(value, label)
    require(result.is_integer(), f"{label}: 需要非负整数。")
    return int(result)


def reject_constant(value: str) -> None:
    raise ValueError(f"JSON 中包含非有限常量：{value}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    require(isinstance(value, dict), f"{path}: 顶层必须是对象。")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_hash(value: Any, label: str) -> None:
    require(isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"{label}: 缺少有效 SHA-256，不能确认配对。")


def validate_metrics(metrics: dict[str, Any], label: str) -> None:
    require(metrics["relative_norm_threshold"] == 1e-10,
            f"{label}: 相对误差无定义阈值必须保持 1e-10。")
    for key in ("mse", "mixed_loss", "macro_mean_relative_l2"):
        number(metrics[key], f"{label}.{key}")
    counts = []
    for field in FIELDS:
        data = metrics["fields"][field]
        for metric in ("absolute_l2", "relative_l2", "reference_l2"):
            values = data[metric]
            for stat in STATS:
                number(values[stat], f"{label}.{field}.{metric}.{stat}")
            defined = integer(values["defined_count"], f"{label}.{field}.{metric}.defined")
            undefined = integer(values["undefined_count"], f"{label}.{field}.{metric}.undefined")
            require(undefined == 0,
                    f"{label}.{field}.{metric}: 存在无定义指标；需先人工处理，"
                    "不能改变分母或比较人群后自动选方向。")
            require(defined > 0, f"{label}.{field}.{metric}: 没有可比较样本。")
            counts.append(defined)
    require(len(set(counts)) == 1, f"{label}: 三场及各指标的样本数不一致。")
    expected = sum(float(metrics["fields"][c]["relative_l2"]["mean"])
                   for c in FIELDS) / 3
    require(close(float(metrics["macro_mean_relative_l2"]), expected),
            f"{label}: 共同主指标不是三场真实相对 L2 均值的算术平均。")
    joint = metrics["joint_relative_le_1pct"]
    total = integer(joint["total_count"], f"{label}.joint.total")
    passed = integer(joint["pass_count"], f"{label}.joint.pass")
    undefined = integer(joint["undefined_count"], f"{label}.joint.undefined")
    require(total == counts[0] and passed <= total and undefined == 0,
            f"{label}: 共同 1% 阈值统计的人群不一致或含无定义样本。")
    require(close(number(joint["fraction_all"], label + ".joint.fraction"), passed / total),
            f"{label}: 共同阈值通过率与计数不一致。")


def metrics_match(a: Any, b: Any, label: str) -> None:
    """Match all initial metrics, including sample counts and null structure."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), f"{label}: 指标字段不一致。")
        for key in a:
            metrics_match(a[key], b[key], f"{label}.{key}")
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), f"{label}: 列表不一致。")
        for index, (left, right) in enumerate(zip(a, b)):
            metrics_match(left, right, f"{label}[{index}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        require(close(number(a, label), number(b, label)), f"{label}: 初始指标不相同。")
    else:
        require(a == b, f"{label}: 初始指标不相同。")


def metric_columns(metrics: dict[str, Any]) -> dict[str, float]:
    result = {
        "validation_mse": float(metrics["mse"]),
        "validation_mixed_loss": float(metrics["mixed_loss"]),
        "validation_macro_mean_relative_l2": float(metrics["macro_mean_relative_l2"]),
    }
    for field in FIELDS:
        for metric in ("absolute_l2", "relative_l2"):
            for stat in STATS:
                result[f"validation_{field}_{metric}_{stat}"] = float(
                    metrics["fields"][field][metric][stat])
    return result


def verify_record(record: dict[str, Any], row: dict[str, Any], label: str) -> None:
    for key, expected in metric_columns(record["metrics"]).items():
        require(close(number(row[key], label + "." + key), expected),
                f"{label}: {key} 与 history.csv 不一致。")


def load_run(directory: Path, variant: str) -> dict[str, Any]:
    require(directory.is_dir(), f"{variant}: 运行目录不存在：{directory}")
    names = ("experiment.json", "completion.json", "initial_validation.json",
             "best_validation.json", "last_validation.json", "history.csv")
    for name in names:
        require((directory / name).is_file(), f"{variant}: 缺少 {name}，拒绝比较未完成运行。")
    experiment = read_json(directory / "experiment.json")
    completion = read_json(directory / "completion.json")
    require(experiment["variant"] == variant, f"--{variant.lower()} 指向的不是 {variant} 组。")
    require(experiment["objective"] == {"A": "mse", "B": "mixed"}[variant],
            f"{variant}: 保存的训练目标不符合 A/B 定义。")
    require(experiment["reset_optimizer"] is True and experiment["reset_scheduler"] is True,
            f"{variant}: 优化器与调度器必须明确重置。")
    require(experiment["resumable"] is False, f"{variant}: 本比较仅接受独立微调模式。")
    require(integer(experiment["source_epoch"], "source_epoch") == 2478,
            f"{variant}: 初始权重不是指定的 epoch 2478。")
    valid_hash(experiment["source_checkpoint_sha256"], "source_checkpoint_sha256")
    valid_hash(experiment["normalizer_sha256"], "normalizer_sha256")
    require(experiment["normalizer_sha256"] == experiment["data_fingerprints"]["normalizer_sha256"],
            f"{variant}: 归一化散列与数据来源记录不同。")
    comparison = experiment["comparison"]
    require(comparison["primary"] == "macro_mean_relative_l2" and comparison["mode"] == "min"
            and comparison["relative_norm_threshold"] == 1e-10 and comparison["use_test"] is False
            and comparison["checkpoint_selection"] == "same_primary_including_step_zero"
            and comparison["undefined_policy"] == "fail_if_any_validation_relative_undefined",
            f"{variant}: 不是事先规定的共同 validation 比较规则。")
    devices = [experiment.get(section, {}).get("device") for section in ("training", "environment")]
    require("cpu" in devices and all(value in (None, "cpu") for value in devices),
            f"{variant}: experiment 必须显式记录 device='cpu'。")
    require(completion["status"] == "completed", f"{variant}: 运行尚未完成。")
    require(completion["test_evaluated"] is False, f"{variant}: 不能使用查看 test 后的配对流程。")
    require(completion["source_checkpoint_unchanged"] is True,
            f"{variant}: 保存的检查没有确认原始 best 权重保持不变。")
    planned = integer(completion["planned_updates"], "planned_updates")
    require(planned > 0 and integer(completion["completed_updates"], "completed_updates") == planned,
            f"{variant}: 实际参数更新数未达到完整预算。")
    require(integer(experiment["training"]["max_updates"], "training.max_updates") == planned,
            f"{variant}: 声明的训练预算与完成记录不一致。")
    for key in ("train_order_sha256", "lr_sequence_sha256"):
        valid_hash(completion[key], key)
    records = {}
    for kind in ("initial", "best", "last"):
        record = read_json(directory / f"{kind}_validation.json")
        record["step"] = integer(record["step"], f"{variant}.{kind}.step")
        validate_metrics(record["metrics"], f"{variant}.{kind}")
        require(record["metrics"]["mix_lambda"] == experiment["loss"]["lambda"] and
                record["metrics"]["delta_normalized"] == experiment["loss"]["delta"],
                f"{variant}.{kind}: 记录的混合损失系数或幅值下限不一致。")
        records[kind] = record
    require(records["initial"]["step"] == 0, f"{variant}: initial 不是 step=0。")
    require(records["last"]["step"] == planned, f"{variant}: last 不是完整更新预算末端。")
    require(records["best"]["step"] == integer(completion["best_step"], "best_step"),
            f"{variant}: completion 与 best_validation 的 best_step 不同。")
    require(close(number(completion["best_validation_score"], "completion.best_validation_score"),
                  float(records["best"]["metrics"]["macro_mean_relative_l2"])),
            f"{variant}: completion 中的 best 分数不一致。")
    expected_count = records["initial"]["metrics"]["joint_relative_le_1pct"]["total_count"]
    for kind in ("best", "last"):
        require(records[kind]["metrics"]["joint_relative_le_1pct"]["total_count"] == expected_count,
                f"{variant}: {kind} 与 initial 的验证人数不同。")
        for field in FIELDS:
            metrics_match(records["initial"]["metrics"]["fields"][field]["reference_l2"],
                          records[kind]["metrics"]["fields"][field]["reference_l2"],
                          f"{variant}.{kind}.{field}.reference_l2")
    with (directory / "history.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = set(metric_columns(records["initial"]["metrics"])) | {
            "step", "epoch", "learning_rate", "next_learning_rate", "train_online_mse",
            "train_online_mixed_loss", "best_step", "best_validation_score"}
        require(required <= columns, f"{variant}: history 缺少列 {sorted(required - columns)}")
        history = list(reader)
    require(bool(history), f"{variant}: history 没有记录。")
    by_step = {}
    previous = -1
    minimum = float(records["initial"]["metrics"]["macro_mean_relative_l2"])
    score_by_step = {0: minimum}
    for row in history:
        step = integer(row["step"], "history.step")
        require(previous < step <= planned, f"{variant}: history 步数重复、逆序或超过预算。")
        previous = step
        by_step[step] = row
        integer(row["epoch"], "history.epoch")
        for key in required - {"step", "epoch", "best_step"}:
            # No update has used a learning rate or yielded training loss at step zero.
            if step == 0 and (key.startswith("train_online_") or key == "learning_rate") and row[key] == "":
                continue
            number(row[key], f"{variant}.history.{step}.{key}")
        score = float(row["validation_macro_mean_relative_l2"])
        for field in FIELDS:
            defined_key = f"validation_{field}_relative_l2_defined_count"
            undefined_key = f"validation_{field}_relative_l2_undefined_count"
            require(defined_key in row and undefined_key in row,
                    f"{variant}: history 必须保留逐场相对误差定义人数。")
            require(integer(row[defined_key], defined_key) == expected_count and
                    integer(row[undefined_key], undefined_key) == 0,
                    f"{variant}: history 的相对误差验证人群发生变化。")
        mean = sum(float(row[f"validation_{field}_relative_l2_mean"]) for field in FIELDS) / 3
        require(close(score, mean), f"{variant}: step={step} 共同指标与逐场均值不一致。")
        score_by_step[step] = score
        minimum = min(minimum, score)
        require(close(float(row["best_validation_score"]), minimum),
                f"{variant}: history 的累计 best_validation_score 不是共同指标的最小值。")
        running_best_step = integer(row["best_step"], "history.best_step")
        require(running_best_step in score_by_step and
                close(score_by_step[running_best_step], minimum),
                f"{variant}: history 的累计 best_step 不对应共同指标最小值。")
    if 0 in by_step:
        verify_record(records["initial"], by_step[0], f"{variant}.initial")
    require(planned in by_step, f"{variant}: history 缺少完整预算的验证记录。")
    interval = integer(experiment["training"]["validation_every_updates"], "validation_every_updates")
    require(interval > 0 and planned % interval == 0,
            f"{variant}: 验证间隔与更新预算不相容。")
    expected_steps = list(range(interval, planned + 1, interval))
    require([step for step in by_step if step != 0] == expected_steps,
            f"{variant}: history 验证时机不符合事先声明的固定间隔。")
    best_step = records["best"]["step"]
    require(best_step in score_by_step, f"{variant}: best_step 不在已记录的验证候选中。")
    require(close(float(records["best"]["metrics"]["macro_mean_relative_l2"]), minimum),
            f"{variant}: best 不是共同验证指标最优候选（包括初始 step=0）。")
    if best_step == 0:
        metrics_match(records["initial"]["metrics"], records["best"]["metrics"], variant + ".best0")
    else:
        verify_record(records["best"], by_step[best_step], variant + ".best")
    verify_record(records["last"], by_step[planned], variant + ".last")
    return {"directory": str(directory), "experiment": experiment, "completion": completion,
            "records": records, "history": history, "artifact_sha256": {
                name: sha256(directory / name) for name in names}}


def change(current: float, reference: float) -> dict[str, Any]:
    delta = current - reference
    return {"current": current, "reference": reference, "delta": delta,
            "relative_change": delta / reference if reference != 0 else None,
            "increased": current > reference,
            "increase_exceeds_numeric_tolerance": current > reference and not close(current, reference)}


def compare_stage(a: dict[str, Any], b: dict[str, Any], initial: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"primary": {
        "name": "macro_mean_relative_l2",
        "A": float(a["macro_mean_relative_l2"]), "B": float(b["macro_mean_relative_l2"]),
        "B_minus_A": float(b["macro_mean_relative_l2"]) - float(a["macro_mean_relative_l2"]),
        "A_vs_initial": change(float(a["macro_mean_relative_l2"]), float(initial["macro_mean_relative_l2"])),
        "B_vs_initial": change(float(b["macro_mean_relative_l2"]), float(initial["macro_mean_relative_l2"]))},
        "absolute_error_review": {}, "joint_relative_le_1pct": {
            "initial": initial["joint_relative_le_1pct"],
            "A": a["joint_relative_le_1pct"], "B": b["joint_relative_le_1pct"]},
        "absolute_increase_flags": []}
    for field in FIELDS:
        result["absolute_error_review"][field] = {}
        for stat in ("mean", "p95", "max"):
            av = float(a["fields"][field]["absolute_l2"][stat])
            bv = float(b["fields"][field]["absolute_l2"][stat])
            iv = float(initial["fields"][field]["absolute_l2"][stat])
            comparisons = {"A_vs_initial": change(av, iv), "B_vs_initial": change(bv, iv),
                           "B_vs_A": change(bv, av)}
            result["absolute_error_review"][field][stat] = comparisons
            for name, comparison in comparisons.items():
                if comparison["increased"]:
                    result["absolute_increase_flags"].append({"field": field, "statistic": stat,
                                                             "comparison": name, **comparison})
    result["note"] = "共同主指标越低越好；绝对误差逐项审查，不据单一相对指标自动宣布全面胜出。"
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = ["# A/B 保存结果对照", "", "仅读取本次配对运行的 JSON/CSV；未进行训练、预测或 test 评估。", "",
             f"共同初始 checkpoint：`{report['source_checkpoint_sha256']}`，epoch 2478。",
             f"两组实际更新预算均为 {report['planned_updates']}；训练样本顺序及学习率序列散列一致。", "",
             "主指标为三场真实相对 L2 均值的算术平均（越低越好）。"
             "delta 仅用于训练损失，不改变这里的真实相对误差分母。", ""]
    for stage, title in (("best", "按共同验证指标选出的 best"), ("last", "完整更新预算的 last（辅助对照）")):
        data = report[stage]
        p = data["primary"]
        sa, sb = report["selected_steps"][stage]["A"], report["selected_steps"][stage]["B"]
        lines += [f"## {title}", "", f"A：step {sa}；B：step {sb}。",
                  f"共同主指标 A={100*p['A']:.6f}%，B={100*p['B']:.6f}%；"
                  f"B−A={100*p['B_minus_A']:+.6f} 个百分点。", "",
                  "| 场/绝对 L2 统计 | 初始 | A | B | A−初始 | B−初始 | B−A |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for field in FIELDS:
            for stat in ("mean", "p95", "max"):
                item = data["absolute_error_review"][field][stat]
                av, bv = item["A_vs_initial"], item["B_vs_initial"]
                lines.append(f"| {field}/{stat} | {av['reference']:.7g} | {av['current']:.7g} | "
                             f"{bv['current']:.7g} | {av['delta']:+.5g} | {bv['delta']:+.5g} | "
                             f"{item['B_vs_A']['delta']:+.5g} |")
        flags = data["absolute_increase_flags"]
        lines += ["", f"绝对误差上升项目：{len(flags)} 项（逐项详见 JSON；包含任何正增量，"
                  "并另标数值容差内的小变化）。" if flags else "本表没有绝对误差上升项目。", ""]
        joint = data["joint_relative_le_1pct"]
        lines.append("三场同时 ≤1%：" + "；".join(
            f"{name} {joint[name]['pass_count']}/{joint[name]['total_count']}"
            for name in ("initial", "A", "B")) + "。")
        lines.append("")
    lines += ["两组 best 步数可以不同：两组都完成相同预算，再从相同验证时机的候选中按同一指标选择。",
              "A 的原 MSE 与 B 的混合训练目标不能直接按原始数值评胜负；"
              "报告使用共同真实相对误差与物理尺度绝对误差。", "",
              "本报告只支持 validation 上的方向判断，不能证明新材料或新初值泛化提升。"
              "最终研究结论仍需新的独立保留材料/初值测试，不移动现有划分。", ""]
    return "\n".join(lines)


def run(a_dir: Path, b_dir: Path) -> Path:
    require(a_dir != b_dir, "A/B 必须是两个显式指定的独立运行目录。")
    a, b = load_run(a_dir, "A"), load_run(b_dir, "B")
    shared = ("pair_signature", "source_checkpoint_sha256", "source_epoch", "model_config",
              "data_fingerprints", "normalizer_sha256", "implementation_sha256", "environment",
              "training", "comparison", "protocol", "loss")
    for key in shared:
        require(a["experiment"][key] == b["experiment"][key], f"配对失败：{key} 不同。")
        require(a["experiment"][key] not in (None, "", {}), f"配对失败：{key} 未记录。")
    for key in ("planned_updates", "completed_updates", "train_order_sha256", "lr_sequence_sha256"):
        require(a["completion"][key] == b["completion"][key], f"配对失败：{key} 不同。")
    ah, bh = a["history"], b["history"]
    require(len(ah) == len(bh), "配对失败：两组记录的验证次数不同。")
    for left, right in zip(ah, bh):
        for key in ("step", "epoch", "learning_rate", "next_learning_rate"):
            if integer(left["step"], "step") == 0 and key == "learning_rate" and left[key] == right[key] == "":
                continue
            require(close(number(left[key], key), number(right[key], key)),
                    f"配对失败：两组验证时机或学习率计划不同（{key}）。")
    initial = a["records"]["initial"]["metrics"]
    metrics_match(initial, b["records"]["initial"]["metrics"], "A/B.initial")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_sha256": a["experiment"]["source_checkpoint_sha256"],
        "pair_signature": a["experiment"]["pair_signature"],
        "planned_updates": a["completion"]["planned_updates"],
        "comparison_config": a["experiment"]["comparison"],
        "validation_only": True, "test_read": False, "model_loaded": False,
        "initial_match_tolerance": {"rtol": RTOL, "atol": ATOL},
        "initial_metrics": initial,
        "runs": {name: {"directory": data["directory"], "artifact_sha256": data["artifact_sha256"],
                        "experiment": data["experiment"], "completion": data["completion"],
                        "validation": data["records"]} for name, data in (("A", a), ("B", b))},
        "selected_steps": {stage: {name: data["records"][stage]["step"] for name, data in (("A", a), ("B", b))}
                           for stage in ("best", "last")},
        "comparison_script_sha256": sha256(Path(__file__).resolve()),
        "best": compare_stage(a["records"]["best"]["metrics"], b["records"]["best"]["metrics"], initial),
        "last": compare_stage(a["records"]["last"]["metrics"], b["records"]["last"]["metrics"], initial),
        "decision": "待人工根据共同验证指标和绝对误差权衡判断；未自动宣布胜出或泛化结论。",
    }
    # Output location is intentionally not configurable: preserve original runs.
    root = HERE / "comparisons"
    require(root.resolve().is_relative_to(HERE), "comparisons 目录不可通过符号链接指向新增目录之外。")
    root.mkdir(exist_ok=True)
    output = root / ("comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6])
    output.mkdir(exist_ok=False)
    with (output / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    (output / "A_B_验证对照.md").write_text(markdown(report), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="只分析已完成 A/B 保存结果；不训练、不预测、不读取 test。")
    parser.add_argument("--a", type=Path, required=True, help="明确指定 A 组已完成的运行目录")
    parser.add_argument("--b", type=Path, required=True, help="明确指定 B 组已完成的运行目录")
    args = parser.parse_args()
    try:
        output = run(args.a.expanduser().resolve(), args.b.expanduser().resolve())
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(2, f"比较未生成：{error}\n")
    print(f"已核对并比较保存的 validation 结果：{output}")


if __name__ == "__main__":
    main()

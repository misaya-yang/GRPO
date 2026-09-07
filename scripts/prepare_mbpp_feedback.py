"""Freeze a hash-selected deterministic MBPP+ subset before actor generation.

Requires official v0.2.0 JSONL and pinned evalplus/data/mbpp.py for input types.
Special-oracle tasks are excluded a priori; output is a subset mechanism study,
not the full EvalPlus benchmark. Reference execution occurs in each test worker.
"""

import argparse
import ast
import json
from pathlib import Path

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
from reward_coupling.bank import digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True)
parser.add_argument("--decoder-source", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--tests-per-task", type=int, default=8)
parser.add_argument("--seed", type=int, default=20260906)
args = parser.parse_args()
if args.tests_per_task < 2:
    raise ValueError("Need >=2 frozen test configurations")
tree = ast.parse(Path(args.decoder_source).read_text())
definition = next(
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mbpp_deserialize_inputs"
)
namespace = {}
exec(
    compile(ast.Module(body=[definition], type_ignores=[]), args.decoder_source, "exec"), namespace
)
deserialize = namespace["mbpp_deserialize_inputs"]
special = {
    "check_str",
    "text_match_three",
    "text_starta_endb",
    "similar_elements",
    "find_char_long",
    "common_in_nested_lists",
    "extract_singly",
    "larg_nnum",
    "intersection_array",
    "find_dissimilar",
    "Diff",
}
excluded, eligible = [], []
raw = [json.loads(s) for s in Path(args.source).read_text().splitlines()]
for task in raw:
    if not isinstance(task["plus_input"], list):
        excluded.append(
            {"id": task["task_id"], "reason": "unsupported_or_missing_plus_input_schema"}
        )
        continue
    if task["entry_point"] in special or task["task_id"] in ("Mbpp/558", "Mbpp/581"):
        excluded.append(
            {"id": task["task_id"], "reason": "predeclared_special_oracle_not_in_subset"}
        )
        continue
    inputs = deserialize(task["task_id"], task["base_input"] + task["plus_input"])
    # Stable hash ordering; duplicates removed before observing any actor output.
    unique = {repr(value): value for value in inputs}
    selected = sorted(unique, key=lambda value: digest([args.seed, task["task_id"], value]))[
        : args.tests_per_task
    ]
    if len(selected) < args.tests_per_task:
        excluded.append({"id": task["task_id"], "reason": "too_few_distinct_test_inputs"})
        continue
    tests = []
    for encoded in selected:
        test = (
            f"__reference_scope={{}}\nexec({task['canonical_solution']!r},__reference_scope)\n"
            f"__actual={task['entry_point']}(*{encoded})\n"
            f"__expected=__reference_scope[{task['entry_point']!r}](*{encoded})\n"
            "try:\n __equal=bool(__actual==__expected)\nexcept (TypeError,ValueError):\n __equal=False\n"
            "if not __equal:\n"
            " import numpy as np\n"
            f" __atol={task['atol']!r}\n"
            " if __atol==0 and isinstance(__expected,float): __atol=1e-6\n"
            " assert __atol>0\n"
            " assert np.shape(__actual)==np.shape(__expected)\n"
            " assert np.allclose(__actual,__expected,rtol=1e-7,atol=__atol)\n"
        )
        tests.append(test)
    eligible.append(
        {
            "prompt_id": task["task_id"],
            "prompt": task["prompt"]
            + "\nReturn only the complete Python function implementation in one Python code block.",
            "reference_code": task["canonical_solution"],
            "tests": tests,
            "test_input_repr": selected,
            "source": f"MbppPlus-v0.2.0_hash_selected_{args.tests_per_task}_test_subset",
            "entry_point": task["entry_point"],
            "atol": task["atol"],
        }
    )
eligible.sort(key=lambda t: digest([args.seed, t["prompt_id"]]))
if len(eligible) < 112:
    raise ValueError("Insufficient supported tasks before generation")
counts = [("Dev", 16), ("C", 32), ("D", 32), ("evaluation", 32)]
offset = 0
for split, count in counts:
    for row in eligible[offset : offset + count]:
        row["split"] = split
    offset += count
for row in eligible[offset:]:
    row["split"] = "train"
write_jsonl(args.output, eligible)
write_json(
    args.output + ".receipt.json",
    {
        "source_sha256": sha256(args.source),
        "decoder_sha256": sha256(args.decoder_source),
        "tasks_sha256": sha256(args.output),
        "seed": args.seed,
        "excluded": excluded,
        "rows": len(eligible),
        "status": "frozen_before_actor_reference_check_required",
        "tests_per_task": args.tests_per_task,
    },
)
print(args.output)

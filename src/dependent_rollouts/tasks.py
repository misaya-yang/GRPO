"""Local JSONL tasks, conservative final-answer parsing, rational AST verifier."""

import ast
import hashlib
import json
import random
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

VERIFIER_VERSION = "final_tag_rational_ast_v1"


def rational_expression(text, numbers=None):
    if len(text) > 256:
        raise ValueError("Expression too long")
    tree = ast.parse(text.strip(), mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("Expression too complex")
    used = []

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) > 10**9:
                raise ValueError("Integer too large")
            used.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            return a / b
        raise ValueError("Only integers, parentheses, +, -, *, / are allowed")

    value = visit(tree.body)
    if numbers is not None and Counter(used) != Counter(numbers):
        raise ValueError("Each supplied number must occur exactly once")
    return value


def verify(response, task, finish_reason):
    """Truncations remain observed failures. Exactly one terminal answer tag."""
    if finish_reason != "eos":
        return {"reward": 0, "category": "truncated"}
    matches = list(re.finditer(r"<answer>([^<>]+)</answer>", response))
    if len(matches) != 1 or response[matches[0].end() :].strip():
        return {"reward": 0, "category": "invalid_format"}
    answer = matches[0].group(1).strip()
    try:
        if task["task_type"] == "arithmetic":
            value = rational_expression(answer, task["numbers"])
        elif task["task_type"] == "numeric":
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)", answer):
                raise ValueError("Expected a complete numeric answer")
            value = Fraction(answer)
        else:
            raise ValueError("Unsupported task type")
        success = value == Fraction(str(task["target"]))
        return {"reward": int(success), "category": "correct" if success else "wrong_value"}
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return {"reward": 0, "category": "invalid_expression"}


def generate_arithmetic(count, split, seed):
    """Distinct seed and expression structure per split; deduplicate by numbers/target."""
    templates = {
        "calibration": lambda a, b, c, d: f"({a}+{b})*({c}+{d})",
        "discovery": lambda a, b, c, d: f"({a}*{b})+({c}*{d})",
        "confirmation": lambda a, b, c, d: f"({a}+{b})*{c}-{d}",
        "train": lambda a, b, c, d: f"({a}*{b}-{c})*{d}",
        "evaluation": lambda a, b, c, d: f"{a}*({b}+{c}*{d})",
    }
    if split not in templates or count < 1:
        raise ValueError("Invalid split or count")
    rng = random.Random(f"{seed}:{split}")
    rows, seen = [], set()
    for _ in range(count * 100):
        numbers = [rng.randint(1, 15) for _ in range(4)]
        expression = templates[split](*numbers)
        target = str(rational_expression(expression))
        key = (tuple(sorted(numbers)), target)
        if key in seen:
            continue
        seen.add(key)
        identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        rows.append(
            {
                "prompt_id": f"arith-{identity}",
                "split": split,
                "task_type": "arithmetic",
                "numbers": numbers,
                "target": target,
                "template_id": split,
                "prompt": f"Use each of {numbers} exactly once with +, -, *, / and parentheses "
                f"to make {target}. You may reason first. End with exactly one "
                "<answer>expression</answer> tag and nothing after it.",
            }
        )
        if len(rows) == count:
            return rows
    raise ValueError("Unable to generate enough unique instances")


def read_tasks(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty task file")
    ids, prompts = set(), set()
    for row in rows:
        if row["prompt_id"] in ids or row["prompt"] in prompts:
            raise ValueError("Duplicate prompt id or text")
        if row["task_type"] not in ("arithmetic", "numeric"):
            raise ValueError("Unsupported task")
        ids.add(row["prompt_id"])
        prompts.add(row["prompt"])
        Fraction(str(row["target"]))
    return rows

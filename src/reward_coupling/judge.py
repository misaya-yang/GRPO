"""Local pinned judge, one response per request; no external API or group context."""

import json
import re
from collections.abc import Mapping

from .bank import digest
from .execute_tests import InfrastructureError


class LocalRubricJudge:
    def __init__(self, config):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not re.fullmatch(r"[0-9a-f]{40}", config.get("revision", "")):
            raise ValueError("Pin local judge revision")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], revision=config["revision"], trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"], revision=config["revision"], trust_remote_code=False
        )
        self.model.to(config["device"]).eval()

    def evaluate(self, text, task):
        import torch

        criteria = task["criteria"]
        request = {"task": task["prompt"], "criteria": criteria, "response": text}
        messages = [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": json.dumps(request)},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if not hasattr(ids, "ndim"):
            ids = torch.tensor(ids)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise InfrastructureError("Judge chat template did not produce one tokenized request")
        ids = ids.to(next(self.model.parameters()).device)
        with torch.no_grad():
            generated = self.model.generate(
                ids, do_sample=False, max_new_tokens=self.config["max_new_tokens"]
            )
        raw = self.tokenizer.decode(generated[0, ids.shape[1] :], skip_special_tokens=True)
        try:
            verdicts = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InfrastructureError(
                "Judge returned invalid JSON; pause without reward substitution"
            ) from exc
        if (
            not isinstance(verdicts, list)
            or len(verdicts) != len(criteria)
            or any(type(v) is not int or v not in (0, 1) for v in verdicts)
        ):
            raise InfrastructureError("Judge criterion coverage/schema failure")
        return {
            "test_verdicts": verdicts,
            "raw_judge_output": raw,
            "judge_request_hash": digest(messages),
            "judge_config": self.config,
            "test_manifest_hash": digest(criteria),
            "response_local": True,
            "private_randomness": "greedy_cached_realization_not_live_judge_expectation",
        }

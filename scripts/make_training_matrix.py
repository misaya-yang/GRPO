"""Write the six matched arms x three seeds; does not run jobs or change GO."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("template")
parser.add_argument("output_dir")
args = parser.parse_args()
template = json.loads(Path(args.template).read_text())
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=False)
for sampler in ("iid", "stratified", "lattice"):
    for arm in ("within", "cross"):
        for seed in (202711, 202712, 202713):
            config = dict(template, sampler=sampler, arm=arm, seed=seed, groups_per_prompt=2)
            with (out / f"{sampler}-{arm}-{seed}.json").open("x") as handle:
                json.dump(config, handle, indent=2)
                handle.write("\n")

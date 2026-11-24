"""Generate the MQL5 EA files for the rules accepted by stage 10 (stage 12). Example:
python scripts/gen_ea.py --run output/run1 [--rules output/run1/10_validation/accepted_rules.json] [--rule R02441] [--config my.yaml]
Outputs: <run>/12_ea/ER_<rule>.mq5, ER_<rule>.plan.json, refused.json (ALL reasons per refused rule), REVIEW_CHECKLIST.md, ea_info.json
The generator refuses a rule it cannot translate exactly (rolling-state features, events, custom indicators, intrabar exits, calendar); it never approximates.
"""

import argparse
import json

from edge_research.codegen.stage import generate_eas
from edge_research.config import load_config
from edge_research.rules import load_rules


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rules", default=None, help="canonical rule file (default: <run>/10_validation/accepted_rules.json)")
    ap.add_argument("--rule", action="append", default=None, help="only this rule id (repeatable)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rules = load_rules(a.rules) if a.rules else None
    info = generate_eas(a.run, load_config(a.config), rules, a.out, a.rule)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

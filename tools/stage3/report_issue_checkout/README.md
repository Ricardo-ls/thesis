# report_issue_checkout

Focused read-only checkout for current Stage 3 report issues raised by the professor.

Run from repository root:

```bash
.venv/bin/python -m tools.stage3.report_issue_checkout.build_checkout
```

Scope:

- inspect existing scripts, data files, tables, and figures
- do not modify methods, models, metrics, reconstruction logic, report text, or existing result files
- write isolated outputs only under `outputs/stage3/report_issue_checkout/`

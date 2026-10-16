# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on this repository. Do not open a public issue.
You can expect a first answer within 7 days. This is a research project maintained by one person,
so there is no bug bounty and only the latest `main` is supported.

## Known risk: model checkpoints are code

`.pt` checkpoints are loaded with PyTorch / Ultralytics, which unpickle them. Loading a checkpoint
can execute arbitrary code. Only pass `--weights` / `--model` files you produced yourself or
downloaded from a source you trust (for example the official Ultralytics release assets).
The same applies to Valohai inputs and to any `.pt` attached to an issue or PR.

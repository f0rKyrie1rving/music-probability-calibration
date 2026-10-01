import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Standalone music probability-calibration benchmark")
    subs = parser.add_subparsers(dest="command", required=True)
    audit = subs.add_parser("audit")
    audit.add_argument("--data", type=Path, required=True)
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--data", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    for name in ["run", "summarize", "verify"]:
        p = subs.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name == "summarize":
            p.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        import json
        from .data import audit_bundle
        print(json.dumps(audit_bundle(args.data)[1], indent=2))
    elif args.command == "freeze":
        from .experiment import freeze
        freeze(args.data, args.out)
    elif args.command == "run":
        from .experiment import run
        run(args.out)
    elif args.command == "summarize":
        from .report import summarize
        summarize(args.out, args.report)
    else:
        from .verify import verify
        verify(args.out)


if __name__ == "__main__":
    main()

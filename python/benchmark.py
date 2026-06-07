from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

try:
    from racket_dynamics.configs import DEFAULT_RACKETS, make_experiment_config
    from racket_dynamics.evaluation.runner import run_experiment
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing dependency: {missing}. Install the benchmark dependencies with "
        "`pip install -r requirements.txt` from the repository root."
    ) from exc


PAPER_METHODS = (
    "parametric_constant",
    "parametric_stateful",
    "gp_parameters_full",
    "mlp_parameters_full",
    "residual_mlp_full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper benchmark on the fixed train/test splits."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=PAPER_METHODS,
        choices=PAPER_METHODS,
        help="Benchmark methods to run. Defaults to the paper comparison set.",
    )
    parser.add_argument(
        "--rackets",
        nargs="+",
        default=DEFAULT_RACKETS,
        help="Racket IDs to evaluate, for example: --rackets 01 04 10.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/benchmark",
        help="Directory for reports and summaries.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also save diagnostic plots. This makes the benchmark slower.",
    )
    return parser.parse_args()


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "racket",
        "train",
        "test",
        "velocity_mae",
        "spin_mae",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(global_rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        "| method | velocity MAE | spin MAE |",
        "| --- | ---: | ---: |",
    ]
    for row in global_rows:
        lines.append(
            "| {method} | {velocity_mean:.4f} +- {velocity_std:.4f} | "
            "{spin_mean:.4f} +- {spin_std:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def format_path_for_config(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots" if args.plots else None

    per_racket_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []

    for method_name in args.methods:
        print(f"running {method_name}")
        config = make_experiment_config(
            method_name=method_name,
            rackets=tuple(args.rackets),
            output_path=str(reports_dir / f"{method_name}.json"),
            plot_dir=None if plots_dir is None else str(plots_dir),
        )
        result = run_experiment(config)
        report = result.report

        global_rows.append(
            {
                "method": report.method_name,
                "velocity_mean": report.mean_velocity_mae,
                "velocity_std": report.std_velocity_mae,
                "spin_mean": report.mean_spin_mae,
                "spin_std": report.std_spin_mae,
            }
        )
        for racket_report in report.rackets:
            per_racket_rows.append(
                {
                    "method": report.method_name,
                    "racket": racket_report.racket_id,
                    "train": racket_report.num_train,
                    "test": racket_report.num_test,
                    "velocity_mae": racket_report.metrics.velocity_mae_mean,
                    "spin_mae": racket_report.metrics.spin_mae_mean,
                }
            )

        print(
            f"  velocity_mae={report.mean_velocity_mae:.4f}+-{report.std_velocity_mae:.4f} "
            f"spin_mae={report.mean_spin_mae:.4f}+-{report.std_spin_mae:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(per_racket_rows, output_dir / "per_racket_metrics.csv")
    write_summary_markdown(global_rows, output_dir / "summary.md")
    (output_dir / "summary.json").write_text(json.dumps(global_rows, indent=2))
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(
            {
                "methods": list(args.methods),
                "rackets": list(args.rackets),
                "split": "precomputed train/test CSV files",
                "reports_dir": format_path_for_config(reports_dir),
                "plots": args.plots,
            },
            indent=2,
        )
    )

    print(f"reports: {reports_dir}")
    print(f"summary: {output_dir / 'summary.md'}")
    print(f"per-racket metrics: {output_dir / 'per_racket_metrics.csv'}")


if __name__ == "__main__":
    main()

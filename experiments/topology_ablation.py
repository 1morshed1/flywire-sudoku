"""Topology ablation: does FlyWire wiring help, or just its degree distribution?

Trains the FlyWire SNN under several topology conditions at a **fixed neuron count and
matched edge budget**, repeated over seeds, and reports move accuracy / solve rate plus
graph statistics for each. The load-bearing comparison is FlyWire vs the
degree-preserving shuffle (PLAN.md §17): if FlyWire wins, structure beyond the degree
sequence matters.

Results are written as JSON to ``results/`` (gitignored) and printed as a table.

Run:
    uv run --no-sync python -m experiments.topology_ablation --config configs/ablation.yaml
    uv run --no-sync python -m experiments.topology_ablation --quick   # fast 4x4 smoke
"""

from __future__ import annotations

import dataclasses
import json
import statistics as pystats
from dataclasses import dataclass, field
from pathlib import Path

from connectome.pipeline import subgraph
from connectome.statistics import summary as graph_summary
from connectome.topology import apply_topology
from training.supervised import TrainConfig, train_supervised


@dataclass
class AblationConfig:
    """What conditions to sweep. ``base`` is the shared training config."""

    base: TrainConfig = field(default_factory=lambda: TrainConfig(model="flywire_snn"))
    topologies: tuple[str, ...] = ("flywire", "shuffled", "random")
    seeds: tuple[int, ...] = (0, 1, 2)
    out_dir: str = "results"


def _graph_stats(cfg: TrainConfig, topology: str, seed: int) -> dict[str, float]:
    """Graph statistics for one condition's mask (for the results table)."""
    sub, _ = subgraph(
        cfg.n_neurons, method=cfg.sample_method, weight=cfg.conn_weight, seed=cfg.seed
    )
    sub = apply_topology(sub, topology, seed=seed)
    return graph_summary(sub)


def run_ablation(cfg: AblationConfig, *, verbose: bool = True) -> list[dict]:
    """Run every (topology, seed) condition; return one result dict per run."""
    results: list[dict] = []
    for topology in cfg.topologies:
        for seed in cfg.seeds:
            run_cfg = dataclasses.replace(
                cfg.base, topology=topology, topology_seed=seed, seed=seed
            )
            metrics = train_supervised(run_cfg, verbose=False)
            stats = _graph_stats(run_cfg, topology, seed)
            row = {
                "topology": topology,
                "seed": seed,
                "move_acc": metrics["move_acc"],
                "solve_rate": metrics["solve_rate"],
                "params": metrics["params"],
                "num_edges": stats["num_edges"],
                "reciprocity": stats["reciprocity"],
                "largest_component_frac": stats["largest_component_frac"],
            }
            results.append(row)
            if verbose:
                print(
                    f"[ablation] {topology:11s} seed={seed} "
                    f"move_acc={row['move_acc']:.3f} solve_rate={row['solve_rate']:.3f} "
                    f"recip={row['reciprocity']:.3f}"
                )
    _write_and_summarize(cfg, results, verbose=verbose)
    return results


def _write_and_summarize(cfg: AblationConfig, results: list[dict], *, verbose: bool) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "topology_ablation.json").write_text(json.dumps(results, indent=2))
    if not verbose:
        return
    print("\n=== topology ablation summary (mean over seeds) ===")
    print(f"{'topology':12s} {'move_acc':>9s} {'solve_rate':>11s} {'reciprocity':>12s}")
    for topology in cfg.topologies:
        rows = [r for r in results if r["topology"] == topology]
        if not rows:
            continue
        ma = pystats.mean(r["move_acc"] for r in rows)
        sr = pystats.mean(r["solve_rate"] for r in rows)
        rc = pystats.mean(r["reciprocity"] for r in rows)
        print(f"{topology:12s} {ma:9.3f} {sr:11.3f} {rc:12.3f}")


def _quick_config() -> AblationConfig:
    """A fast CPU-friendly 4x4 smoke sweep."""
    base = TrainConfig(
        model="flywire_snn",
        side=4,
        n_neurons=500,
        T=10,
        n_train=800,
        n_val=200,
        difficulty="easy",
        epochs=15,
        batch_size=64,
        device="cpu",
    )
    return AblationConfig(base=base, topologies=("flywire", "shuffled", "random"), seeds=(0,))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    from training.supervised import _load_yaml_config

    parser = argparse.ArgumentParser(description="FlyWire topology ablation.")
    parser.add_argument("--config", type=str, default=None, help="base TrainConfig YAML")
    parser.add_argument("--quick", action="store_true", help="fast 4x4 smoke sweep")
    args = parser.parse_args()

    if args.quick:
        ablation = _quick_config()
    elif args.config:
        ablation = AblationConfig(base=_load_yaml_config(args.config))
    else:
        ablation = AblationConfig()
    run_ablation(ablation)

# eval/stability.py

import numpy as np


def compute_stability(metrics_log: list, growth_steps: list, window: int = 50):
    """
    For each spawn/growth event:
    1. |delta loss| at t -> t+1
    2. Windowed trend slope +-window steps around event
    3. Spike detection: post-event loss > 2.5x pre-event average

    Returns stability_report list and a summary dict.
    """
    loss_by_step = {m["step"]: m["loss"] for m in metrics_log}
    all_steps = sorted(loss_by_step.keys())
    report = []

    for t in growth_steps:
        loss_before = loss_by_step.get(t - 1) or loss_by_step.get(t)
        loss_at = loss_by_step.get(t)
        loss_after = loss_by_step.get(t + 1)

        delta = abs(loss_after - loss_before) if (loss_after and loss_before) else None

        win_steps = [s for s in all_steps if t - window <= s <= t + window]
        win_losses = [loss_by_step[s] for s in win_steps]

        slope = None
        if len(win_losses) >= 5:
            x = np.arange(len(win_losses))
            slope = float(np.polyfit(x, win_losses, 1)[0])

        pre_losses = [loss_by_step[s] for s in win_steps if s < t]
        post_losses = [loss_by_step[s] for s in win_steps if s > t]
        pre_avg = np.mean(pre_losses) if pre_losses else None
        spike = bool(pre_avg and post_losses and max(post_losses[:5]) > 2.5 * pre_avg)

        report.append({
            "spawn_step": t,
            "loss_before": loss_before,
            "loss_at_event": loss_at,
            "loss_after": loss_after,
            "delta": delta,
            "window_slope": slope,
            "spike_detected": spike
        })

    n_spikes = sum(1 for r in report if r["spike_detected"])
    summary = {
        "n_events": len(report),
        "n_spikes": n_spikes,
        "max_delta": max((r["delta"] for r in report if r["delta"] is not None), default=None),
        "all_stable": n_spikes == 0
    }

    print("\n=== Stability Report ===")
    for r in report:
        status = "SPIKE" if r["spike_detected"] else "STABLE"
        print(f"  Step {r['spawn_step']:5d}: |delta|={r['delta']:.6f if r['delta'] else 'N/A'}, "
              f"slope={r['window_slope']:.6f if r['window_slope'] else 'N/A'} -> {status}")
    print(f"Summary: {n_spikes}/{len(report)} spikes | "
          f"{'CLAIM 2 PASS' if n_spikes == 0 else 'CLAIM 2 FAIL'}")

    return report, summary

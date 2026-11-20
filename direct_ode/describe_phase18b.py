#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptive summary generator for Phase 18B outputs (no simulation/fitting).
Writes phase18b_describe.txt with E1 envelope coverage, E2 status quo drift
(growth vs decline), E3 hindsight vaccination comparison, and scenario effect sizes.
"""

import argparse
import json
import os
import pandas as pd


def describe(in_dir="phase18b_projections_v2", out_file="phase18b_describe.txt"):
    with open(os.path.join(in_dir, "criteria.json"), "r", encoding="utf-8") as f:
        crit = json.load(f)
    e1 = pd.read_csv(os.path.join(in_dir, "envelope_e1.csv"))
    e2 = pd.read_csv(os.path.join(in_dir, "drift_e2.csv"))
    e3 = pd.read_csv(os.path.join(in_dir, "hindsight_vax_e3.csv"))
    summ = pd.read_csv(os.path.join(in_dir, "projections_summary.csv"))

    lines = []
    lines.append("=== PHASE 18B DESCRIPTIVE SUMMARY ===")
    lines.append(
        f"E1 overall met: {crit['E1_met']} "
        f"({crit['E1_counties_passed']}/3 counties >= 50% w1-4; 2 required)"
    )
    lines.append(
        f"Repro check: max_rel_p0_vs_saved_h1_4 = {crit['repro_check']['max_rel_p0_vs_saved_h1_4']:.3e} | "
        f"max_abs_pop_error = {crit['repro_check']['max_abs_pop_error']:.3e}"
    )

    lines.append("\n--- 1. E1: Envelope Coverage (all chains) ---")
    lines.append(e1.to_string(index=False))

    lines.append("\n--- 2. E2: Status Quo Drift (Primary Chain L3B delay=1, realized_phase) ---")
    e2_p = e2[
        (e2["model"] == "L3B")
        & (e2["delay"] == 1)
        & (e2["phase_definition"] == "realized_phase")
    ]
    lines.append(
        e2_p[
            [
                "county",
                "h",
                "phase",
                "n_pairs",
                "median_log_p0_over_obs",
                "median_ratio_p0_over_obs",
                "median_ratio_naive_over_obs",
            ]
        ].to_string(index=False)
    )

    lines.append("\n--- 3. E3: Hindsight Vaccination vs Status Quo P0 (Primary Chain L3B delay=1) ---")
    e3_p = e3[(e3["model"] == "L3B") & (e3["delay"] == 1)]
    lines.append(e3_p.to_string(index=False))

    lines.append("\n--- 4. Scenario Effects over 8 Weeks vs P0 (Medians across origins with doses, L3B delay=1) ---")
    s_p = summ[
        (summ["model"] == "L3B")
        & (summ["delay"] == 1)
        & (summ["scenario"] != "P0")
        & (summ["has_doses"] == True)
    ]
    med_s = (
        s_p.groupby(["county", "scenario"], as_index=False)[
            [
                "pct_total_adm_vs_P0",
                "pct_peak_adm_vs_P0",
                "peak_shift_weeks_vs_P0",
                "pct_total_inf_vs_P0",
            ]
        ]
        .median()
    )
    lines.append(med_s.to_string(index=False))

    text = "\n".join(lines) + "\n"
    print(text)
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="phase18b_projections_v2")
    parser.add_argument("--out", default="phase18b_describe.txt")
    args = parser.parse_args()
    describe(in_dir=args.in_dir, out_file=args.out)
"""Module 4: 5-Year Dynamic LBO and 5,000-Iteration Cholesky Monte Carlo Simulator with Minimum Cash Buffer Rule."""

from typing import Any, Dict, List
import numpy as np

from src.api.schemas import is_en_lang


class LBORiskSimulator:
    """
    Dynamic 5-Year Leveraged Buyout (LBO) Financial Model and 5,000-Iteration Correlated Monte Carlo
    Risk Simulator enforcing the Minimum Operating Cash Buffer rule during debt cash sweep.
    """

    def __init__(self, n_iterations: int = 5000, random_state: int = 42):
        self.n_iterations = n_iterations
        self.random_state = random_state

    def _run_deterministic_schedule(
        self,
        rev_0: float,
        ebitda_0: float,
        entry_ev: float,
        senior_debt_0: float,
        sub_debt_0: float,
        cash_0: float,
        base_growth: float,
        base_margin: float,
        senior_rate: float,
        sub_rate: float,
        capex_pct_rev: float,
        nwc_pct_delta_rev: float,
        tax_rate: float,
        min_cash_buffer_pct_rev: float,
    ) -> List[Dict[str, float]]:
        rev = rev_0
        senior_debt = senior_debt_0
        sub_debt = sub_debt_0
        cash = max(cash_0, rev_0 * min_cash_buffer_pct_rev, 10.0)

        schedule: List[Dict[str, float]] = []
        for yr in range(1, 6):
            prev_rev = rev
            rev = prev_rev * (1.0 + base_growth)
            ebitda = rev * base_margin
            depreciation = rev * (capex_pct_rev * 0.78)
            ebit = ebitda - depreciation

            interest_exp = senior_debt * senior_rate + sub_debt * sub_rate
            ebt = ebit - interest_exp
            taxes = max(0.0, ebt * tax_rate)
            net_income = ebt - taxes

            capex = rev * capex_pct_rev
            delta_nwc = max(0.0, (rev - prev_rev) * nwc_pct_delta_rev)
            fcf = net_income + depreciation - capex - delta_nwc

            # Minimum Operating Cash Buffer Rule: cash is NEVER swept below min_cash_buffer
            min_cash_req = max(rev * min_cash_buffer_pct_rev, 10.0)
            pre_sweep_cash = cash + fcf

            if pre_sweep_cash > min_cash_req:
                sweepable_cash = pre_sweep_cash - min_cash_req
                paid_senior = min(senior_debt, sweepable_cash)
                senior_debt -= paid_senior
                rem_sweep = sweepable_cash - paid_senior
                paid_sub = min(sub_debt, rem_sweep)
                sub_debt -= paid_sub
                total_debt_paydown = paid_senior + paid_sub
                cash = min_cash_req
            else:
                # Cash buffer preservation: no debt sweep when cash falls below operating buffer
                total_debt_paydown = 0.0
                shortfall = min_cash_req - pre_sweep_cash
                senior_debt += shortfall  # Revolver draw to preserve minimum operating cash buffer
                cash = min_cash_req

            total_ending_debt = senior_debt + sub_debt
            net_debt = total_ending_debt - cash

            schedule.append(
                {
                    "year": yr,
                    "revenue_usd_m": round(rev, 1),
                    "ebitda_usd_m": round(ebitda, 1),
                    "interest_expense_usd_m": round(interest_exp, 1),
                    "free_cash_flow_usd_m": round(fcf, 1),
                    "min_cash_buffer_usd_m": round(min_cash_req, 1),
                    "debt_paydown_usd_m": round(total_debt_paydown, 1),
                    "ending_senior_debt_usd_m": round(senior_debt, 1),
                    "ending_sub_debt_usd_m": round(sub_debt, 1),
                    "ending_cash_usd_m": round(cash, 1),
                    "net_leverage_turns": round(net_debt / max(ebitda, 1.0), 2),
                }
            )
        return schedule

    def _compute_deterministic_irr_moic(
        self,
        rev_0: float,
        ebitda_0: float,
        entry_ev: float,
        leverage_turns: float,
        exit_multiple: float,
        senior_share: float,
        senior_rate: float,
        sub_rate: float,
        base_growth: float,
        base_margin: float,
        min_cash_buffer_pct_rev: float,
    ) -> Dict[str, float]:
        total_debt_0 = min(ebitda_0 * leverage_turns, entry_ev * 0.85)
        senior_0 = total_debt_0 * senior_share
        sub_0 = total_debt_0 * (1.0 - senior_share)
        min_cash_0 = max(rev_0 * min_cash_buffer_pct_rev, 10.0)
        sponsor_equity_0 = max(entry_ev - total_debt_0 + min_cash_0, entry_ev * 0.15)

        sched = self._run_deterministic_schedule(
            rev_0=rev_0,
            ebitda_0=ebitda_0,
            entry_ev=entry_ev,
            senior_debt_0=senior_0,
            sub_debt_0=sub_0,
            cash_0=min_cash_0,
            base_growth=base_growth,
            base_margin=base_margin,
            senior_rate=senior_rate,
            sub_rate=sub_rate,
            capex_pct_rev=0.048,
            nwc_pct_delta_rev=0.06,
            tax_rate=0.23,
            min_cash_buffer_pct_rev=min_cash_buffer_pct_rev,
        )
        yr5 = sched[-1]
        exit_ev = yr5["ebitda_usd_m"] * exit_multiple
        ending_net_debt = yr5["ending_senior_debt_usd_m"] + yr5["ending_sub_debt_usd_m"] - yr5["ending_cash_usd_m"]
        exit_equity = max(exit_ev - ending_net_debt, 0.0)
        moic = exit_equity / max(sponsor_equity_0, 1.0)
        irr_pct = ((max(moic, 0.05) ** (1.0 / 5.0)) - 1.0) * 100.0
        return {"irr_pct": round(float(irr_pct), 1), "moic": round(float(moic), 2)}

    def simulate(
        self,
        target: Dict[str, Any],
        entry_ev_usd_m: float = 0.0,
        total_debt_ebitda_turns: float = 4.5,
        senior_debt_share: float = 0.70,
        senior_interest_rate: float = 0.065,
        subordinated_interest_rate: float = 0.108,
        exit_ev_ebitda_multiple: float = 0.0,
        min_cash_buffer_pct_rev: float = 0.04,
        lang: str = "tr",
    ) -> Dict[str, Any]:
        rev_0 = max(float(target["revenue"]), 20.0)
        ebitda_0 = max(float(target["ebitda"]), 4.0)
        base_margin = float(np.clip(ebitda_0 / rev_0, 0.05, 0.65))
        base_growth = float(np.clip(float(target.get("rev_growth_1y", 0.09)), -0.10, 0.35))

        if entry_ev_usd_m <= 0:
            entry_ev_usd_m = max(float(target["enterprise_value"]) * 1.24, ebitda_0 * 6.0)
        if exit_ev_ebitda_multiple <= 0:
            exit_ev_ebitda_multiple = round(max(entry_ev_usd_m / ebitda_0, 6.5), 1)

        total_debt_0 = min(ebitda_0 * total_debt_ebitda_turns, entry_ev_usd_m * 0.85)
        senior_0 = total_debt_0 * senior_debt_share
        sub_0 = total_debt_0 * (1.0 - senior_debt_share)
        min_cash_0 = max(rev_0 * min_cash_buffer_pct_rev, 10.0)
        sponsor_equity_0 = max(entry_ev_usd_m - total_debt_0 + min_cash_0, entry_ev_usd_m * 0.15)

        capex_pct_rev = 0.048
        nwc_pct_delta_rev = 0.06
        tax_rate = 0.23

        # 1. 5-Year Deterministic Base Schedule
        schedule = self._run_deterministic_schedule(
            rev_0=rev_0,
            ebitda_0=ebitda_0,
            entry_ev=entry_ev_usd_m,
            senior_debt_0=senior_0,
            sub_debt_0=sub_0,
            cash_0=min_cash_0,
            base_growth=base_growth,
            base_margin=base_margin,
            senior_rate=senior_interest_rate,
            sub_rate=subordinated_interest_rate,
            capex_pct_rev=capex_pct_rev,
            nwc_pct_delta_rev=nwc_pct_delta_rev,
            tax_rate=tax_rate,
            min_cash_buffer_pct_rev=min_cash_buffer_pct_rev,
        )

        # 2. Vectorized 5,000-Iteration Cholesky-Correlated Monte Carlo Simulation
        n_sims = self.n_iterations
        rng = np.random.default_rng(self.random_state)

        # Correlation matrix across 4 risk factors: [rev_growth, margin_shock, rate_shock, exit_mult_shock]
        corr_mat = np.array(
            [
                [1.00, 0.45, -0.25, 0.40],
                [0.45, 1.00, -0.30, 0.38],
                [-0.25, -0.30, 1.00, -0.35],
                [0.40, 0.38, -0.35, 1.00],
            ]
        )
        chol = np.linalg.cholesky(corr_mat)

        rev_vec = np.full(n_sims, rev_0, dtype=float)
        senior_vec = np.full(n_sims, senior_0, dtype=float)
        sub_vec = np.full(n_sims, sub_0, dtype=float)
        cash_vec = np.full(n_sims, min_cash_0, dtype=float)
        crunch_flag = np.zeros(n_sims, dtype=bool)

        exit_mult_shocks = np.zeros(n_sims, dtype=float)

        for yr in range(5):
            z_raw = rng.standard_normal((n_sims, 4))
            z_corr = z_raw @ chol.T
            if yr == 4:
                exit_mult_shocks = z_corr[:, 3]

            g_t = np.clip(base_growth + 0.055 * z_corr[:, 0], -0.28, 0.48)
            m_t = np.clip(base_margin + 0.026 * z_corr[:, 1], 0.03, 0.65)
            r_shock = np.clip(0.011 * z_corr[:, 2], -0.02, 0.045)

            prev_rev_vec = rev_vec
            rev_vec = prev_rev_vec * (1.0 + g_t)
            ebitda_vec = rev_vec * m_t
            dep_vec = rev_vec * (capex_pct_rev * 0.78)
            ebit_vec = ebitda_vec - dep_vec

            int_vec = senior_vec * np.maximum(senior_interest_rate + r_shock, 0.02) + sub_vec * np.maximum(
                subordinated_interest_rate + 0.6 * r_shock, 0.04
            )
            ebt_vec = ebit_vec - int_vec
            tax_vec = np.maximum(0.0, ebt_vec * tax_rate)
            ni_vec = ebt_vec - tax_vec

            capex_vec = rev_vec * capex_pct_rev
            dnwc_vec = np.maximum(0.0, (rev_vec - prev_rev_vec) * nwc_pct_delta_rev)
            fcf_vec = ni_vec + dep_vec - capex_vec - dnwc_vec

            # Mandatory debt service check & Minimum Cash Buffer enforcement
            mandatory_amort = 0.03 * senior_0
            min_cash_req_vec = np.maximum(rev_vec * min_cash_buffer_pct_rev, 10.0)
            pre_sweep_vec = cash_vec + fcf_vec

            # A path experiences a cash crunch / debt service failure if pre-sweep cash falls below min_cash + mandatory amort
            yr_crunch = (pre_sweep_vec - mandatory_amort < min_cash_req_vec) | (ebitda_vec - capex_vec < int_vec)
            crunch_flag = crunch_flag | yr_crunch

            excess_cash = np.maximum(0.0, pre_sweep_vec - min_cash_req_vec)
            paid_sen = np.minimum(senior_vec, excess_cash)
            senior_vec -= paid_sen
            rem_excess = excess_cash - paid_sen
            paid_sub = np.minimum(sub_vec, rem_excess)
            sub_vec -= paid_sub

            shortfall_vec = np.maximum(0.0, min_cash_req_vec - pre_sweep_vec)
            senior_vec += shortfall_vec
            cash_vec = min_cash_req_vec

        sim_exit_mult = np.clip(exit_ev_ebitda_multiple + 1.55 * exit_mult_shocks, 3.8, 24.0)
        exit_ev_vec = ebitda_vec * sim_exit_mult
        ending_net_debt_vec = senior_vec + sub_vec - cash_vec
        exit_equity_vec = np.maximum(exit_ev_vec - ending_net_debt_vec, 0.0)

        moic_vec = exit_equity_vec / max(sponsor_equity_0, 1.0)
        irr_vec = ((np.maximum(moic_vec, 0.04) ** (1.0 / 5.0)) - 1.0) * 100.0

        # Build 20-bin IRR distribution histogram
        clipped_irr = np.clip(irr_vec, -25.0, 55.0)
        counts, bin_edges = np.histogram(clipped_irr, bins=20, range=(-20.0, 50.0))
        histogram = []
        for b_idx in range(len(counts)):
            b_mid = 0.5 * (bin_edges[b_idx] + bin_edges[b_idx + 1])
            histogram.append(
                {
                    "bin_start": round(float(bin_edges[b_idx]), 1),
                    "bin_end": round(float(bin_edges[b_idx + 1]), 1),
                    "bin_center": round(float(b_mid), 1),
                    "count": int(counts[b_idx]),
                    "frequency_pct": round(float(counts[b_idx] / n_sims * 100.0), 2),
                }
            )

        # 3. 5x5 Sensitivity Matrix (Entry Leverage Turns x Exit EV/EBITDA Multiple)
        lev_grid = [
            round(max(total_debt_ebitda_turns - 1.5, 2.0), 1),
            round(max(total_debt_ebitda_turns - 0.75, 2.5), 1),
            round(total_debt_ebitda_turns, 1),
            round(total_debt_ebitda_turns + 0.75, 1),
            round(total_debt_ebitda_turns + 1.5, 1),
        ]
        mult_grid = [
            round(max(exit_ev_ebitda_multiple - 2.0, 4.5), 1),
            round(max(exit_ev_ebitda_multiple - 1.0, 5.5), 1),
            round(exit_ev_ebitda_multiple, 1),
            round(exit_ev_ebitda_multiple + 1.0, 1),
            round(exit_ev_ebitda_multiple + 2.0, 1),
        ]

        cells = []
        for l_val in lev_grid:
            for m_val in mult_grid:
                cell_res = self._compute_deterministic_irr_moic(
                    rev_0=rev_0,
                    ebitda_0=ebitda_0,
                    entry_ev=entry_ev_usd_m,
                    leverage_turns=l_val,
                    exit_multiple=m_val,
                    senior_share=senior_debt_share,
                    senior_rate=senior_interest_rate,
                    sub_rate=subordinated_interest_rate,
                    base_growth=base_growth,
                    base_margin=base_margin,
                    min_cash_buffer_pct_rev=min_cash_buffer_pct_rev,
                )
                cells.append(
                    {
                        "leverage_turns": l_val,
                        "exit_multiple": m_val,
                        "irr_pct": cell_res["irr_pct"],
                        "moic": cell_res["moic"],
                    }
                )

        return {
            "target_ticker": target["ticker"],
            "target_name": target["company_name"],
            "n_iterations": n_sims,
            "capital_structure": {
                "entry_ev_usd_m": round(entry_ev_usd_m, 1),
                "total_debt_usd_m": round(total_debt_0, 1),
                "senior_debt_usd_m": round(senior_0, 1),
                "subordinated_debt_usd_m": round(sub_0, 1),
                "sponsor_equity_usd_m": round(sponsor_equity_0, 1),
                "initial_min_cash_buffer_usd_m": round(min_cash_0, 1),
                "debt_to_ev_pct": round(total_debt_0 / max(entry_ev_usd_m, 1.0) * 100.0, 1),
                "equity_to_ev_pct": round(sponsor_equity_0 / max(entry_ev_usd_m, 1.0) * 100.0, 1),
            },
            "summary": {
                "expected_irr_pct": round(float(np.mean(irr_vec)), 1),
                "median_irr_pct": round(float(np.median(irr_vec)), 1),
                "p10_irr_pct": round(float(np.percentile(irr_vec, 10)), 1),
                "p90_irr_pct": round(float(np.percentile(irr_vec, 90)), 1),
                "expected_moic": round(float(np.mean(moic_vec)), 2),
                "median_moic": round(float(np.median(moic_vec)), 2),
                "p10_moic": round(float(np.percentile(moic_vec, 10)), 2),
                "p90_moic": round(float(np.percentile(moic_vec, 90)), 2),
                "prob_above_20pct_irr": round(float(np.mean(irr_vec >= 20.0) * 100.0), 1),
                "prob_capital_loss_pct": round(float(np.mean(moic_vec < 1.0) * 100.0), 1),
                "stress_default_probability_pct": round(float(np.mean(crunch_flag) * 100.0), 1),
            },
            "projection_schedule": schedule,
            "irr_histogram": histogram,
            "sensitivity_matrix": {
                "leverage_axes": lev_grid,
                "exit_multiple_axes": mult_grid,
                "cells": cells,
            },
        }

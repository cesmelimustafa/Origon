"""Module 4: 5-Year Dynamic LBO and 5,000-Iteration Cholesky Monte Carlo Simulator with Minimum Cash Buffer Rule."""

from typing import Any, Dict, List, Union
import numpy as np

from src.api.schemas import is_en_lang

TX_FEE_PCT_EV = 0.0225


def deductible_interest(
    interest_exp: Union[float, np.ndarray],
    ebitda: Union[float, np.ndarray],
    total_debt: Union[float, np.ndarray],
    equity: Union[float, np.ndarray],
    region: str = "US (NYSE/NASDAQ)",
) -> Union[float, np.ndarray]:
    """
    Tax-deductible interest expense under regional corporate tax rules:
    - US (IRC Section 163(j), EBITDA base permanently restored by OBBBA 2025) & EU (ATAD):
      deductible interest is capped at 30% of EBITDA.
    - Turkey (KVK Art. 11/1-i financing expense restriction):
      when total debt exceeds equity, 10% of the financing expense attributable to the excess
      debt share is disallowed (KKEG).
    """
    is_turkey = "turkey" in str(region).lower() or "bist" in str(region).lower() or "türkiye" in str(region).lower()
    if is_turkey:
        eq_pos = np.maximum(equity, 0.0)
        excess_debt = np.maximum(total_debt - eq_pos, 0.0)
        excess_share = np.where(total_debt > 1e-6, excess_debt / np.maximum(total_debt, 1e-6), 0.0)
        disallowed = interest_exp * excess_share * 0.10
        res = np.maximum(interest_exp - disallowed, 0.0)
    else:
        cap = np.maximum(0.30 * ebitda, 0.0)
        res = np.minimum(interest_exp, cap)

    if np.ndim(interest_exp) == 0 and np.ndim(ebitda) == 0:
        return float(res)
    return res


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
        sponsor_equity_0: float = 0.0,
        region: str = "US (NYSE/NASDAQ)",
    ) -> List[Dict[str, float]]:
        rev = rev_0
        senior_debt = senior_debt_0
        sub_debt = sub_debt_0
        cash = max(cash_0, rev_0 * min_cash_buffer_pct_rev, 10.0)
        book_equity = max(sponsor_equity_0, entry_ev - senior_debt_0 - sub_debt_0, 10.0)

        schedule: List[Dict[str, float]] = []
        for yr in range(1, 6):
            prev_rev = rev
            rev = prev_rev * (1.0 + base_growth)
            ebitda = rev * base_margin
            depreciation = rev * (capex_pct_rev * 0.78)
            ebit = ebitda - depreciation

            capex = rev * capex_pct_rev
            rev_delta = rev - prev_rev
            delta_nwc = (rev_delta * nwc_pct_delta_rev) if rev_delta >= 0.0 else (0.5 * rev_delta * nwc_pct_delta_rev)
            min_cash_req = max(rev * min_cash_buffer_pct_rev, 10.0)

            beg_senior = senior_debt
            beg_sub = sub_debt
            end_senior = beg_senior
            end_sub = beg_sub
            end_cash = cash
            interest_exp = beg_senior * senior_rate + beg_sub * sub_rate
            ded_int = interest_exp
            net_income = 0.0
            fcf = 0.0
            total_debt_paydown = 0.0

            # Fixed-point iteration resolves average-debt interest circularity (Rosenbaum & Pearl)
            for _ in range(5):
                avg_senior = 0.5 * (beg_senior + end_senior)
                avg_sub = 0.5 * (beg_sub + end_sub)
                interest_exp = avg_senior * senior_rate + avg_sub * sub_rate
                ded_int = float(
                    deductible_interest(interest_exp, ebitda, beg_senior + beg_sub, book_equity, region)
                )
                taxable_inc = max(0.0, ebit - ded_int)
                taxes = taxable_inc * tax_rate
                net_income = ebit - interest_exp - taxes
                fcf = net_income + depreciation - capex - delta_nwc

                # Minimum Operating Cash Buffer Rule: cash is NEVER swept below min_cash_buffer
                pre_sweep_cash = cash + fcf
                if pre_sweep_cash > min_cash_req:
                    sweepable_cash = pre_sweep_cash - min_cash_req
                    paid_senior = min(beg_senior, sweepable_cash)
                    end_senior = beg_senior - paid_senior
                    rem_sweep = sweepable_cash - paid_senior
                    paid_sub = min(beg_sub, rem_sweep)
                    end_sub = beg_sub - paid_sub
                    total_debt_paydown = paid_senior + paid_sub
                    end_cash = min_cash_req
                else:
                    total_debt_paydown = 0.0
                    shortfall = min_cash_req - pre_sweep_cash
                    end_senior = beg_senior + shortfall
                    end_sub = beg_sub
                    end_cash = min_cash_req

            senior_debt = end_senior
            sub_debt = end_sub
            cash = end_cash
            book_equity = max(book_equity + net_income, 1.0)

            total_ending_debt = senior_debt + sub_debt
            net_debt = total_ending_debt - cash
            icr = ebitda / max(interest_exp, 0.1)

            schedule.append(
                {
                    "year": yr,
                    "revenue_usd_m": round(rev, 1),
                    "ebitda_usd_m": round(ebitda, 1),
                    "interest_expense_usd_m": round(interest_exp, 1),
                    "deductible_interest_usd_m": round(ded_int, 1),
                    "interest_coverage": round(icr, 2),
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
        region: str = "US (NYSE/NASDAQ)",
    ) -> Dict[str, float]:
        tx_fees = entry_ev * TX_FEE_PCT_EV
        total_debt_0 = min(ebitda_0 * leverage_turns, entry_ev * 0.85)
        senior_0 = total_debt_0 * senior_share
        sub_0 = total_debt_0 * (1.0 - senior_share)
        min_cash_0 = max(rev_0 * min_cash_buffer_pct_rev, 10.0)
        sponsor_equity_0 = max(entry_ev + tx_fees + min_cash_0 - total_debt_0, entry_ev * 0.15)

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
            sponsor_equity_0=sponsor_equity_0,
            region=region,
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
        region_code = str(target.get("region_code", "US (NYSE/NASDAQ)"))
        rev_0 = max(float(target["revenue"]), 20.0)
        ebitda_0 = max(float(target["ebitda"]), 4.0)
        base_margin = float(np.clip(ebitda_0 / rev_0, 0.05, 0.65))
        base_growth = float(np.clip(float(target.get("rev_growth_1y", 0.09)), -0.10, 0.35))

        if entry_ev_usd_m <= 0:
            entry_ev_usd_m = max(float(target["enterprise_value"]) * 1.24, ebitda_0 * 6.0)
        if exit_ev_ebitda_multiple <= 0:
            exit_ev_ebitda_multiple = round(max(entry_ev_usd_m / ebitda_0, 6.5), 1)

        tx_fees_0 = entry_ev_usd_m * TX_FEE_PCT_EV
        total_debt_0 = min(ebitda_0 * total_debt_ebitda_turns, entry_ev_usd_m * 0.85)
        senior_0 = total_debt_0 * senior_debt_share
        sub_0 = total_debt_0 * (1.0 - senior_debt_share)
        min_cash_0 = max(rev_0 * min_cash_buffer_pct_rev, 10.0)
        sponsor_equity_0 = max(entry_ev_usd_m + tx_fees_0 + min_cash_0 - total_debt_0, entry_ev_usd_m * 0.15)

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
            sponsor_equity_0=sponsor_equity_0,
            region=region_code,
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
        equity_vec = np.full(n_sims, sponsor_equity_0, dtype=float)
        crunch_flag = np.zeros(n_sims, dtype=bool)
        covenant_breach_flag = np.zeros(n_sims, dtype=bool)
        max_lev_covenant = max(total_debt_ebitda_turns + 1.5, 6.5)

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

            capex_vec = rev_vec * capex_pct_rev
            rev_delta_vec = rev_vec - prev_rev_vec
            dnwc_vec = np.where(
                rev_delta_vec >= 0.0,
                rev_delta_vec * nwc_pct_delta_rev,
                0.5 * rev_delta_vec * nwc_pct_delta_rev,
            )
            min_cash_req_vec = np.maximum(rev_vec * min_cash_buffer_pct_rev, 10.0)
            eff_sen_rate = np.maximum(senior_interest_rate + r_shock, 0.02)
            eff_sub_rate = np.maximum(subordinated_interest_rate + 0.6 * r_shock, 0.04)

            beg_sen_vec = senior_vec
            beg_sub_vec = sub_vec
            end_sen_vec = beg_sen_vec.copy()
            end_sub_vec = beg_sub_vec.copy()

            # 2-pass fixed point for vectorized average-debt interest & regional tax shield cap
            for _ in range(2):
                avg_sen_vec = 0.5 * (beg_sen_vec + end_sen_vec)
                avg_sub_vec = 0.5 * (beg_sub_vec + end_sub_vec)
                int_vec = avg_sen_vec * eff_sen_rate + avg_sub_vec * eff_sub_rate
                ded_int_vec = deductible_interest(
                    int_vec, ebitda_vec, beg_sen_vec + beg_sub_vec, equity_vec, region_code
                )
                tax_vec = np.maximum(0.0, (ebit_vec - ded_int_vec) * tax_rate)
                ni_vec = ebit_vec - int_vec - tax_vec
                fcf_vec = ni_vec + dep_vec - capex_vec - dnwc_vec

                pre_sweep_vec = cash_vec + fcf_vec
                excess_cash = np.maximum(0.0, pre_sweep_vec - min_cash_req_vec)
                paid_sen = np.minimum(beg_sen_vec, excess_cash)
                rem_excess = excess_cash - paid_sen
                paid_sub = np.minimum(beg_sub_vec, rem_excess)
                shortfall_vec = np.maximum(0.0, min_cash_req_vec - pre_sweep_vec)
                end_sen_vec = beg_sen_vec - paid_sen + shortfall_vec
                end_sub_vec = beg_sub_vec - paid_sub

            # Mandatory debt service check & maintenance covenants (ICR < 1.5x or Net Leverage > covenant)
            mandatory_amort = 0.03 * senior_0
            yr_crunch = (pre_sweep_vec - mandatory_amort < min_cash_req_vec) | (ebitda_vec - capex_vec < int_vec)
            crunch_flag = crunch_flag | yr_crunch

            net_lev_vec = (end_sen_vec + end_sub_vec - min_cash_req_vec) / np.maximum(ebitda_vec, 1.0)
            icr_vec = ebitda_vec / np.maximum(int_vec, 0.1)
            covenant_breach_flag = covenant_breach_flag | (icr_vec < 1.50) | (net_lev_vec > max_lev_covenant)

            senior_vec = end_sen_vec
            sub_vec = end_sub_vec
            cash_vec = min_cash_req_vec
            equity_vec = np.maximum(equity_vec + ni_vec, 1.0)

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
                    region=region_code,
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
                "transaction_fees_usd_m": round(tx_fees_0, 1),
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
                "covenant_breach_probability_pct": round(float(np.mean(covenant_breach_flag) * 100.0), 1),
            },
            "projection_schedule": schedule,
            "irr_histogram": histogram,
            "sensitivity_matrix": {
                "leverage_axes": lev_grid,
                "exit_multiple_axes": mult_grid,
                "cells": cells,
            },
        }

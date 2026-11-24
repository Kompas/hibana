"""agent_eval's summary and paired-comparison maths on fixed numbers, against tools/lossy_gate.py's ci_mean (CPU)."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_eval  # noqa: E402
from mtp_capture import use_repo  # noqa: E402

use_repo()
from lossy_gate import ci_mean  # noqa: E402

# (id, source, model, output tokens, rounds, decode seconds, proposed, accepted, relaxed) for the stock arm, and the
# tuned arm's (rounds, decode seconds, accepted) on the same prompts and outputs
STOCK = [("p1", "pi", "qwen-trial117", 257, 80, 2.50, 400, 176, 10), ("p2", "pi", "kimi", 129, 50, 1.40, 200, 78, 4),
         ("p3", "claude", "x", 257, 64, 2.10, 320, 192, 0), ("p4", "pi", "qwen-trial117", 65, 20, 0.60, 80, 44, 2),
         ("p5", "codex", "y", 257, 100, 3.00, 500, 156, 7), ("p6", "claude", "x", 200, 61, 1.90, 300, 138, 1)]
TUNED = {"p1": (64, 2.10, 192), "p2": (45, 1.30, 83), "p3": (62, 2.05, 194), "p4": (16, 0.50, 48),
         "p5": (90, 2.80, 166), "p6": (60, 1.92, 139)}


def row(pid, src, model, n, rounds, dec, prop, acc, rel):
    return {"id": pid, "source": src, "model": model, "prompt_tokens": 1000, "output_tokens": n, "rounds": rounds,
            "proposed": prop, "accepted": acc, "emitted": n - 1, "relaxed": rel, "tok_per_round": (n - 1) / rounds,
            "accept_rate": acc / prop, "relaxed_share": rel / n, "decode_s": dec, "prefill_s": 0.1,
            "decode_tps": (n - 1) / dec, "round_ms": 1000 * dec / rounds, "accepted_hist": {"0": 1, "3": rounds - 1},
            "output_sha256": f"h{pid}"}


def arms():
    base = [row(*r) for r in STOCK]
    cand = []
    for r in STOCK:
        rounds, dec, acc = TUNED[r[0]]
        cand.append(row(r[0], r[1], r[2], r[3], rounds, dec, r[6], acc, r[8]))
    cand[2]["output_sha256"] = "different"
    ident = {"prompts_sha256": "x", "accept": [3, 2.0], "max_new": 256, "w3_identity": "w", "profile": "fast"}
    return ({"label": "stock", "arm": {**ident, "mtp_identity": "a"}, "prompts": base},
            {"label": "tuned", "arm": {**ident, "mtp_identity": "b"}, "prompts": cand})


def test_ci_mean_fixed_numbers():
    m, lo, hi = ci_mean([0.1, 0.2, 0.3, 0.4])
    assert m == pytest.approx(0.25) and lo <= m <= hi and lo >= 0.1 and hi <= 0.4


def test_compare_matches_lossy_gate_bootstrap():
    base, cand = arms()
    cmp = agent_eval.compare(base, cand)
    assert cmp["paired"] == 6 and cmp["warnings"] == []
    groups = {"all": [r[0] for r in STOCK], "pi": ["p1", "p2", "p4"], "pi/qwen-trial117": ["p1", "p4"], "pi/other": ["p2"],
              "claude": ["p3", "p6"], "codex": ["p5"]}
    assert set(cmp["groups"]) == set(groups)
    b = {r["id"]: r for r in base["prompts"]}
    c = {r["id"]: r for r in cand["prompts"]}
    for g, ids in groups.items():
        for key in ("tok_per_round", "decode_tps", "accept_rate"):
            logs = [math.log(c[i][key] / b[i][key]) for i in ids]
            m, lo, hi = ci_mean(logs)  # the gate's interval, by prompt
            got = cmp["groups"][g][key]
            assert got["n"] == len(ids)
            assert got["ratio"] == pytest.approx(math.exp(m))
            assert got["ci95"] == pytest.approx([math.exp(lo), math.exp(hi)])
    # the ratio is the geometric mean of the per-prompt ratios: p1 tokens/round 256/64 over 256/80 = 1.25, p4 1.25
    assert cmp["groups"]["pi/qwen-trial117"]["tok_per_round"]["ratio"] == pytest.approx(1.25)
    assert cmp["groups"]["pi/other"]["tok_per_round"]["ratio"] == pytest.approx(50 / 45)
    # pooled: sum(n - 1) / sum(rounds), cand over base
    all_b = sum(r[3] - 1 for r in STOCK) / sum(r[4] for r in STOCK)
    all_c = sum(r[3] - 1 for r in STOCK) / sum(TUNED[r[0]][0] for r in STOCK)
    assert cmp["groups"]["all"]["pooled"]["tok_per_round"] == pytest.approx(all_c / all_b)
    assert cmp["groups"]["all"]["same_output"] == 5
    table = agent_eval.compare_table(cmp)
    assert "| pi/qwen-trial117 | 2 | 1.250 [1.250, 1.250]" in table


def test_compare_warns_on_mismatched_arms():
    base, cand = arms()
    cand["arm"]["accept"] = [1, 0.0]
    cand["arm"]["mtp_identity"] = "a"
    cand["prompts"] = cand["prompts"][:4]
    w = agent_eval.compare(base, cand)["warnings"]
    assert any("accept" in x for x in w) and any("same mtp" in x for x in w) and any("4 paired" in x for x in w)


def test_summarize_pooled_numbers():
    base, _ = arms()
    s = agent_eval.summarize(base["prompts"])
    pi = s["pi"]
    assert pi["prompts"] == 3 and pi["rounds"] == 150
    assert pi["tok_per_round"] == pytest.approx((256 + 128 + 64) / 150)
    assert pi["accept_rate"] == pytest.approx((176 + 78 + 44) / (400 + 200 + 80))
    assert pi["relaxed_share"] == pytest.approx(16 / (257 + 129 + 65))
    assert pi["decode_tps"] == pytest.approx((256 + 128 + 64) / (2.5 + 1.4 + 0.6))
    assert pi["decode_tps_mean"] == pytest.approx((256 / 2.5 + 128 / 1.4 + 64 / 0.6) / 3)
    assert s["pi/qwen-trial117"]["prompts"] == 2 and s["pi/other"]["prompts"] == 1
    assert "claude/other" not in s and s["all"]["prompts"] == 6
    assert s["pi"]["accepted_hist"] == {"0": 3, "3": 147}


def test_prompt_row_from_request():
    r = SimpleNamespace(output_ids=list(range(101)), spec_rounds=25, spec_proposed=150, spec_accepted=75, spec_emitted=100,
                        spec_relaxed=5, finish_reason="length")
    p = {"id": "q", "source": "pi", "model": "qwen-trial117", "prompt_tokens": [1] * 900}
    out = agent_eval.prompt_row(p, r, 1.0, 0.2, [(6, 3)] * 20 + [(4, 2)] * 5)
    assert out["tok_per_round"] == 4.0 and out["decode_tps"] == 100.0 and out["accept_rate"] == 0.5
    assert out["round_ms"] == 40.0 and out["relaxed_share"] == pytest.approx(5 / 101)
    assert out["accepted_hist"] == {"3": 20, "2": 5} and out["depth_hist"] == {"6": 20, "4": 5}
    assert out["prompt_tokens"] == 900 and "prompt_tokens" in out and len(out["output_sha256"]) == 16

"""Real (LLM) Nexus on Kaggle: the full Ollama pipeline over the benchmark series grid.

Same code path as scripts/run_water_nexus.py (arm "ollama" = the 4-call gemma4:26b pipeline,
temperature 0, seed 42, thinking off), generalised from water to every resource in
region_series_dump.json: electricity (95 zones), carbon (95 zones), water (24 basins).
Holdout = the last 12 months of each series, as in the earlier cached benchmark, so the output can be
scored against the cached xLSTM / TimesFM 2.5 / Chronos-2 forecasts.

Nothing is lost if a session ends early:
  1. every series is appended to OUT/nexus_<resource>_ollama.csv the moment it finishes;
  2. TIME BUDGET: the run stops by itself before NEXUS_MAX_HOURS (counted from NEXUS_START, set by
     the notebook's first cell), so a committed run ends normally and Kaggle saves its output;
  3. CHECKPOINTS: every NEXUS_CKPT_MIN minutes (and at the end) the CSVs are pushed as a new version
     of a PRIVATE Kaggle dataset <you>/nexus-results-checkpoint, if KAGGLE_USERNAME and KAGGLE_KEY
     are set (the notebook reads them from Kaggle Secrets). That copy survives a hard kill;
  4. RESUME: at start, CSVs from every attached input (an earlier version's output and/or the
     checkpoint dataset) are merged with what is already in OUT; a series counts as done only
     if all 12 of its rows are present, so a series cut off mid-write is simply redone.

    python run_nexus_kaggle.py check                 # consistency check: 3 water basins vs local run
    python run_nexus_kaggle.py electricity carbon    # the main run (water already exists locally)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
OUT = os.environ.get("NEXUS_OUT", "/kaggle/working")
HOLDOUT = 12
DUMP = os.path.join(HERE, "work", "region_series_dump.json")
LOCAL_WATER = os.path.join(HERE, "work", "local_cpu_water_ollama.csv")   # the CPU run, for the check
START = float(os.environ.get("NEXUS_START", time.time()))
MAX_HOURS = float(os.environ.get("NEXUS_MAX_HOURS", "11"))          # Kaggle commit limit is 12 h
CKPT_MIN = float(os.environ.get("NEXUS_CKPT_MIN", "30"))
CKPT_SLUG = "nexus-results-checkpoint"
CKPT_DIR = os.environ.get("NEXUS_CKPT_DIR", "/tmp/nexus_ckpt")
INPUT = os.environ.get("NEXUS_INPUT", "/kaggle/input")
REPEATS = int(os.environ.get("NEXUS_REPEATS", "1"))   # LLM Nexus is not repeatable on GPU: run each series N times


def log(msg: str):
    print(msg, flush=True)


def load(resource: str) -> dict[str, pd.Series]:
    d = json.load(open(DUMP))[resource]
    # raw index as scored by the notebook (water is mid-month with gaps: no frequency attached)
    return {rid: pd.Series(p["values"], index=pd.to_datetime(p["index"])) for rid, p in d.items()}


def _complete(d: pd.DataFrame) -> pd.DataFrame:
    """Keep only (series, repeat) units with all HOLDOUT rows; one copy of each (first seen wins).
    Files written before repeats existed have no `rep` column and count as repeat 1."""
    if d.empty:
        return d
    if "rep" not in d.columns:
        d = d.assign(rep=1)
    d = d.drop_duplicates(["region_id", "rep", "timestamp"], keep="first")
    n = d.groupby(["region_id", "rep"])["timestamp"].transform("size")
    return d[n == HOLDOUT]


def merge_resume(resource: str) -> str:
    """OUT file := complete series from OUT itself plus every attached input copy."""
    path = os.path.join(OUT, f"nexus_{resource}_ollama.csv")
    srcs = ([path] if os.path.exists(path) else []) + \
        sorted(glob.glob(f"{INPUT}/**/nexus_{resource}_ollama.csv", recursive=True))
    parts = []
    for f in srcs:
        try:
            parts.append(pd.read_csv(f))
        except Exception as e:
            log(f"  skip unreadable {f}: {e}")
    if parts:
        before = {f: p.region_id.nunique() for f, p in zip(srcs, parts)}
        m = _complete(pd.concat(parts, ignore_index=True))
        m.to_csv(path, index=False)
        log(f"resume {resource}: {len(m.groupby(['region_id', 'rep']))} complete series x repeat units from {before}")
    return path


def done_units(path: str) -> set:
    """{(region_id, rep)} with all HOLDOUT rows."""
    try:
        if not os.path.exists(path):
            return set()
        c = _complete(pd.read_csv(path))
        return set(zip(c["region_id"], c["rep"].astype(int)))
    except Exception:
        return set()


def done(path: str, reps: int | None = None) -> set:
    """Series whose repeats 1..reps are all complete."""
    reps = REPEATS if reps is None else reps
    u = done_units(path)
    return {r for r, _ in u if all((r, k) in u for k in range(1, reps + 1))}


def checkpoint(final: bool = False):
    """Push the result CSVs as a new version of a private Kaggle dataset. Never raises."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        if final:
            log("checkpoint: KAGGLE_USERNAME/KAGGLE_KEY not set -> results only in the notebook output")
        return
    try:
        os.makedirs(CKPT_DIR, exist_ok=True)
        files = glob.glob(os.path.join(OUT, "nexus_*_ollama.csv"))
        if not files:
            return
        for f in files:
            _complete(pd.read_csv(f)).to_csv(os.path.join(CKPT_DIR, os.path.basename(f)), index=False)
        json.dump({"title": CKPT_SLUG, "id": f"{user}/{CKPT_SLUG}", "licenses": [{"name": "CC0-1.0"}]},
                  open(os.path.join(CKPT_DIR, "dataset-metadata.json"), "w"))
        msg = f"checkpoint {time.strftime('%Y-%m-%d %H:%M')}{' final' if final else ''}"
        r = subprocess.run(["kaggle", "datasets", "version", "-p", CKPT_DIR, "-m", msg],
                           capture_output=True, text=True, timeout=600)
        out = (r.stdout + r.stderr).lower()
        if r.returncode != 0 and ("not found" in out or "404" in out):
            r = subprocess.run(["kaggle", "datasets", "create", "-p", CKPT_DIR],   # private by default
                               capture_output=True, text=True, timeout=600)
        ok = r.returncode == 0
        log(f"checkpoint -> {user}/{CKPT_SLUG}: {'ok' if ok else 'FAILED (will retry next time)'} "
            f"{'' if ok else (r.stdout + r.stderr).strip()[-300:]}")
    except Exception as e:
        log(f"checkpoint failed (will retry next time): {type(e).__name__}: {e}")


def hours_used() -> float:
    return (time.time() - START) / 3600


def run(resource: str, only: list[str] | None = None, path: str | None = None, reps: int | None = None) -> bool:
    """Returns False if it stopped early for the time budget. Work unit = one (series, repeat)."""
    from nexus_local import make_nexus_forecast
    import nexus_ollama as NO
    reps = REPEATS if reps is None else reps
    series = load(resource)
    path = path or os.path.join(OUT, f"nexus_{resource}_ollama.csv")
    fn = make_nexus_forecast("ollama")
    have = done_units(path)
    todo = [(r, k) for r in sorted(series) if only is None or r in only
            for k in range(1, reps + 1) if (r, k) not in have]
    log(f"\n=== {resource}: {len(series)} series x {reps} repeat(s), {len(todo)} units to go -> {path} ===")
    t_all, last_ckpt, secs = time.time(), time.time(), []
    for i, (rid, rep) in enumerate(todo, 1):
        typical = float(np.median(secs)) if secs else 0.0
        if hours_used() + 1.5 * typical / 3600 > MAX_HOURS:
            log(f"TIME BUDGET: {hours_used():.2f} h used of {MAX_HOURS} h; stopping cleanly with "
                f"{len(todo) - i + 1} {resource} units left. Re-run to continue.")
            return False
        s = series[rid]
        train, test = s.iloc[:-HOLDOUT], s.iloc[-HOLDOUT:]
        t0 = time.time()
        fc, tag = fn(train, HOLDOUT)
        dt = time.time() - t0
        secs.append(dt)
        pd.DataFrame({"region_id": rid, "rep": rep, "resource": resource, "arm": "ollama", "tag": tag,
                      "timestamp": test.index, "value": np.asarray(fc, float), "actual": test.values,
                      "seconds": round(dt, 1)}).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        rmse = float(np.sqrt(np.mean((np.asarray(fc, float) - test.values) ** 2)))
        eta = (time.time() - t_all) / i * (len(todo) - i) / 3600
        log(f"[{resource}] {i}/{len(todo)} {rid[:30]:<30} rep {rep} {dt:6.1f}s tag={tag:<14} rmse={rmse:10.3f} "
            f"eta={eta:4.1f}h used={hours_used():4.1f}h retries={NO.RETRY_STATS}")
        if time.time() - last_ckpt > CKPT_MIN * 60:
            checkpoint(); last_ckpt = time.time()
    log(f"=== {resource} COMPLETE ===")
    return True


def check():
    """Re-run 3 water basins already done on the local CPU; report the RMSE difference.
    temperature 0 + seed 42 on both, but GPU and CPU kernels are not bit-identical, so small
    differences are expected; a large one means the setup differs (model, options, prompt)."""
    local = pd.read_csv(LOCAL_WATER)
    basins = sorted(local.region_id.unique())[:3]
    path = os.path.join(OUT, "check_water_ollama.csv")
    run("water", only=basins, path=path, reps=1)
    k = pd.read_csv(path)
    rm = lambda d: d.groupby("region_id").apply(lambda g: float(np.sqrt(np.mean((g.value - g.actual) ** 2))),
                                                include_groups=False)
    cmp = pd.DataFrame({"rmse_local_cpu": rm(local[local.region_id.isin(basins)]), "rmse_kaggle_gpu": rm(k)})
    cmp["seconds_local"] = local.groupby("region_id").seconds.first()
    cmp["seconds_kaggle"] = k.groupby("region_id").seconds.first()
    log("\nCONSISTENCY CHECK (same 3 basins, same model and settings):\n" + cmp.round(2).to_string())
    cmp.to_csv(os.path.join(OUT, "check_summary.csv"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:] or ["electricity", "carbon"]
    if args == ["check"]:
        check()
        sys.exit(0)
    log(f"budget {MAX_HOURS} h, {hours_used():.2f} h already used; checkpoint every {CKPT_MIN:.0f} min; "
        f"{REPEATS} repeat(s) per series")
    finished = True
    for r in args:
        merge_resume(r)
    for r in args:
        if not run(r):
            finished = False
            break
    checkpoint(final=True)
    for r in args:
        p = os.path.join(OUT, f"nexus_{r}_ollama.csv")
        log(f"{r}: {len(done(p))}/{len(load(r))} series complete")
    log("ALL DONE" if finished else "STOPPED FOR TIME BUDGET: commit again with this output attached to continue")

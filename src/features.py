"""
Shared feature engineering for the NFL Run-vs-Pass predictor (DSC 148).

This module is the single source of truth for how raw nflfastR play-by-play
rows are turned into model features. It is imported by both the modeling
notebook (notebooks/02_modeling.ipynb) and the demo app (demo/app.py) so that
training and serving use *identical* logic (no train/serve skew).

Two feature groups are defined so we can run a clean ablation study:
  * SITUATION_FEATURES  -- pure game-state info a coordinator knows well before
                           the offense even lines up (down, distance, score,
                           clock, field position, Vegas lines).
  * FORMATION_FEATURES  -- pre-snap formation cues that are only revealed once
                           the offense lines up (shotgun, no-huddle). These are
                           legitimate (known before the snap) but very strong,
                           so we quantify their lift separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw columns we need to pull off the parquet files before engineering.
RAW_COLUMNS = [
    "play_type",
    "season",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "qtr",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "goal_to_go",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "wp",
    "vegas_wp",
    "spread_line",
    "total_line",
    "posteam_type",
    "shotgun",
    "no_huddle",
    "posteam",
    "defteam",
]

# Pure game-situation features (the "fair" decision-prediction task).
SITUATION_FEATURES = [
    # one-hot of down (pass-rate is non-monotonic across downs)
    "down_1", "down_2", "down_3", "down_4",
    # raw context
    "ydstogo",
    "yardline_100",
    "score_differential",
    "qtr",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "goal_to_go",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "wp",
    "vegas_wp",
    "spread_line",
    "total_line",
    "is_home",
    # engineered interactions / football heuristics
    "is_two_minute",
    "is_short_yardage",
    "is_long_yardage",
    "third_and_long",
    "third_and_short",
    "red_zone",
    "trailing",
    "leading",
    "trailing_late",
    "leading_late",
    "score_time_pressure",
]

# Pre-snap formation cues (added on top of SITUATION_FEATURES in the ablation).
FORMATION_FEATURES = ["shotgun", "no_huddle"]

FULL_FEATURES = SITUATION_FEATURES + FORMATION_FEATURES

# Team identity, encoded as leakage-safe historical pass rates (see fit_team_rates).
TEAM_FEATURES = ["posteam_pass_rate", "defteam_pass_rate_faced"]

TARGET = "is_pass"  # 1 = pass, 0 = run


def fit_team_rates(df: pd.DataFrame) -> dict:
    """Historical pass rates per team, computed on TRAINING rows only.

    Returns the offense's pass rate, the defense's pass-rate-faced, and a global
    fallback. Fitting on training data only and applying the result to later seasons
    keeps this a leakage-safe form of target encoding.
    """
    p = (df["play_type"] == "pass").astype(int)
    return {
        "posteam": p.groupby(df["posteam"]).mean().to_dict(),
        "defteam": p.groupby(df["defteam"]).mean().to_dict(),
        "global": float(p.mean()),
    }


def add_team_features(df: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """Map fitted team rates onto a frame; unseen teams fall back to the global rate."""
    df = df.copy()
    df["posteam_pass_rate"] = df["posteam"].map(rates["posteam"]).fillna(rates["global"])
    df["defteam_pass_rate_faced"] = df["defteam"].map(rates["defteam"]).fillna(rates["global"])
    return df


def load_seasons(paths: list[str]) -> pd.DataFrame:
    """Read one or more play-by-play parquet files and keep only run/pass plays."""
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["play_type"].isin(["run", "pass"])].copy()
    keep = [c for c in RAW_COLUMNS if c in df.columns]
    return df[keep].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no `down` (a handful of edge-case snaps) and median-impute
    any remaining stray NaNs in the numeric context columns."""
    df = df.copy()
    df = df[df["down"].notna()].copy()
    num_cols = [
        "ydstogo", "yardline_100", "score_differential", "qtr",
        "half_seconds_remaining", "game_seconds_remaining",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "wp", "vegas_wp", "spread_line", "total_line",
    ]
    for c in num_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived feature columns. Returns a new DataFrame that contains the
    engineered columns plus `season` and the binary target `is_pass`.

    The same function is used at train and serve time. For the demo app, callers
    can pass a single-row DataFrame built from user inputs.
    """
    df = df.copy()

    # ---- target ----
    if "play_type" in df.columns:
        df[TARGET] = (df["play_type"] == "pass").astype(int)

    # ---- down one-hot (1..4) ----
    down = df["down"].astype(int)
    for d in (1, 2, 3, 4):
        df[f"down_{d}"] = (down == d).astype(int)

    # ---- home/away ----
    if "posteam_type" in df.columns:
        df["is_home"] = (df["posteam_type"] == "home").astype(int)
    else:
        df["is_home"] = 0

    # ---- football heuristics / interactions ----
    df["is_two_minute"] = (df["half_seconds_remaining"] <= 120).astype(int)
    df["is_short_yardage"] = (df["ydstogo"] <= 2).astype(int)
    df["is_long_yardage"] = (df["ydstogo"] >= 7).astype(int)
    df["third_and_long"] = ((down == 3) & (df["ydstogo"] >= 7)).astype(int)
    df["third_and_short"] = ((down == 3) & (df["ydstogo"] <= 2)).astype(int)
    df["red_zone"] = (df["yardline_100"] <= 20).astype(int)
    df["trailing"] = (df["score_differential"] < 0).astype(int)
    df["leading"] = (df["score_differential"] > 0).astype(int)
    df["trailing_late"] = ((df["score_differential"] < 0) & (df["qtr"] >= 4)).astype(int)
    df["leading_late"] = ((df["score_differential"] > 0) & (df["qtr"] >= 4)).astype(int)
    # urgency: point deficit per remaining minute (large & negative => must pass)
    df["score_time_pressure"] = df["score_differential"] / (
        df["game_seconds_remaining"] / 60.0 + 1.0
    )

    # make sure formation cues exist and are numeric (for serving/edge cases)
    for c in FORMATION_FEATURES:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)

    return df


def build_xy(df: pd.DataFrame, include_formation: bool = False):
    """Run clean + engineer and return (X, y, season) ready for modeling.

    Parameters
    ----------
    include_formation : bool
        If True, X includes the pre-snap formation cues (shotgun, no_huddle).
    """
    work = engineer(clean(df))
    cols = FULL_FEATURES if include_formation else SITUATION_FEATURES
    X = work[cols].copy()
    y = work[TARGET].copy() if TARGET in work.columns else None
    season = work["season"].copy() if "season" in work.columns else None
    return X, y, season

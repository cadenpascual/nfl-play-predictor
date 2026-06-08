"""
NFL Run vs Pass Predictor, interactive demo (DSC 148)

A small Streamlit UI on top of the models saved by notebooks/02_modeling.ipynb.
Enter a game situation and the app shows the predicted play call and pass
probability, using the exact feature-engineering pipeline from src/features.py
(no train/serve skew).

Two prediction points are available:
  * Before lineup (deployed model): uses only pre-lineup game state (down, distance,
    score, clock, field position, betting market). This is the submitted model.
  * After lineup: additionally uses the pre-snap formation cues (shotgun, no-huddle).
    More accurate, but that information is only available once the offense lines up.

Run with:
    streamlit run demo/app.py
"""
import os
import sys

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import features as F  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "lgbm_runpass.joblib")

st.set_page_config(page_title="NFL Run vs Pass Predictor", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def quarter_clock_to_seconds(qtr, minutes, seconds):
    """Convert a quarter and clock into nflfastR half/game seconds-remaining."""
    secs_left_in_qtr = minutes * 60 + seconds
    if qtr in (1, 3):
        half_remaining = 900 + secs_left_in_qtr
    else:
        half_remaining = secs_left_in_qtr
    elapsed = (qtr - 1) * 900 + (900 - secs_left_in_qtr)
    game_remaining = max(0, 3600 - elapsed)
    return float(half_remaining), float(game_remaining)

def draw_field(yardline_100, ydstogo):
    """Draws a full 2D football field with distinct colored endzones, LOS, and 1st down line."""
    # Set the outer figure background to black
    fig, ax = plt.subplots(figsize=(11, 2.5), facecolor='black')
    
    # Full NFL field dimensions including both 10-yard endzones (-10 to 110)
    ax.set_xlim(-10, 110)
    ax.set_ylim(0, 53.3)
    ax.axis('off')

    # 1. Draw the main playing field (Dark Forest Green)
    field = patches.Rectangle((0, 0), 100, 53.3, linewidth=0, facecolor='#1b4314', zorder=1)
    ax.add_patch(field)

    # 2. Draw the Left Endzone (Sleek Dark Charcoal)
    left_endzone = patches.Rectangle((-10, 0), 10, 53.3, linewidth=0, facecolor='#252525', zorder=1)
    ax.add_patch(left_endzone)

    # 3. Draw the Right Endzone (Sleek Dark Charcoal)
    right_endzone = patches.Rectangle((100, 0), 10, 53.3, linewidth=0, facecolor='#252525', zorder=1)
    ax.add_patch(right_endzone)

    # Draw the boundary outline around the entire field + endzones
    full_border = patches.Rectangle((-10, 0), 120, 53.3, linewidth=2, edgecolor='white', facecolor='none', zorder=2)
    ax.add_patch(full_border)

    # Draw the internal yard lines (every 10 yards)
    for i in range(10, 100, 10):
        ax.axvline(i, color='white', alpha=0.3, linewidth=1.5, zorder=2)
        yard_num = i if i <= 50 else 100 - i
        ax.text(i, 2, str(yard_num), color='white', alpha=0.5, ha='center', va='bottom', fontsize=10, zorder=2)
        ax.text(i, 51.3, str(yard_num), color='white', alpha=0.5, ha='center', va='top', fontsize=10, rotation=180, zorder=2)

    # Calculate Lines
    # Left side (0) is own goal line, Right side (100) is opponent goal line[cite: 1]
    los = 100 - yardline_100 
    first_down = min(100, los + ydstogo) # Caps at the goal line for goal-to-go situations[cite: 1]
    
    # Draw the 1st Down Line (Bright Yellow - TV Broadcast style)
    ax.axvline(first_down, color='#FFFF00', linewidth=3, zorder=3)
    
    # Draw the Line of Scrimmage (Bright Blue)
    ax.axvline(los, color='#0066FF', linewidth=3, zorder=3) 
    
    # Draw a Direction Arrow driving toward opponent endzone
    arrow_length = min(10, first_down - los + 2) if ydstogo > 0 else 5[cite: 1]
    ax.annotate('', xy=(los + arrow_length, 26.65), xytext=(los, 26.65),
                arrowprops=dict(facecolor='white', edgecolor='none', width=3, headwidth=10), zorder=3)

    # Draw the football on top of the arrow
    ax.plot(los, 26.65, marker='D', color='#6e3b22', markersize=8, markeredgecolor='white', zorder=4)
    
    # Endzone text labels (Centered perfectly inside the colored endzone boxes)
    ax.text(-5, 26.65, "OWN\nEnd\nZone", color='white', ha='center', va='center', fontweight='bold', alpha=0.6, fontsize=12, zorder=2)
    ax.text(105, 26.65, "OPP\nEnd\nZone", color='white', ha='center', va='center', fontweight='bold', alpha=0.6, fontsize=12, zorder=2)

    plt.tight_layout()
    return fig

def main():
    st.title("NFL Run vs. Pass Predictor")
    st.caption(
        "Predicts whether an NFL offense will **run** or **pass** before it lines up, from "
        "the game situation. Deployed model: tuned LightGBM trained on 2022-2024 play-by-play."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model artifact not found. Run `notebooks/02_modeling.ipynb` first to "
            "create `models/lgbm_runpass.joblib`."
        )
        st.stop()

    art = load_artifact()
    deployed_model = art["model"]
    formation_model = art.get("formation_model", deployed_model)

    with st.sidebar:
        mode = st.radio(
            "Prediction point",
            ["Before lineup (situation only)", "After lineup (+ formation)"],
            index=0,
            help="The deployed model predicts before the offense lines up, so it uses only "
                 "game state. The after-lineup option also sees the pre-snap formation "
                 "(shotgun, no-huddle): more accurate, but not available before the lineup.",
        )
        after_lineup = mode.startswith("After")
        st.divider()

        st.header("Game situation")
        down = st.selectbox("Down", [1, 2, 3, 4], index=0)
        ydstogo = st.slider("Yards to go", 1, 30, 10)
        yardline_100 = st.slider("Yards from opponent end zone", 1, 99, 65,
                                 help="1 = goal line, 99 = backed up to own end zone")
        goal_to_go = st.checkbox("Goal-to-go", value=(yardline_100 <= ydstogo))

        st.divider()
        qtr = st.selectbox("Quarter", [1, 2, 3, 4, 5], index=0,
                           format_func=lambda q: "OT" if q == 5 else f"Q{q}")
        col_m, col_s = st.columns(2)
        minutes = col_m.number_input("Min left in qtr", 0, 15, 12)
        seconds = col_s.number_input("Sec left in qtr", 0, 59, 0)
        score_differential = st.slider("Score differential (offense minus defense)",
                                       -28, 28, 0)

        st.divider()
        posteam_type = st.radio("Offense is", ["home", "away"], horizontal=True)
        pto = st.slider("Offense timeouts left", 0, 3, 3)
        dto = st.slider("Defense timeouts left", 0, 3, 3)

        shotgun = no_huddle = 0
        if after_lineup:
            st.divider()
            st.subheader("Pre-snap formation")
            shotgun = int(st.checkbox("Shotgun", value=False))
            no_huddle = int(st.checkbox("No-huddle", value=False))

        st.divider()
        st.subheader("Vegas market (optional)")
        win_prob = st.slider("Pre-snap win probability (offense)", 0.0, 1.0, 0.5, 0.01)
        spread_line = st.slider("Spread (home line)", -20.0, 20.0, 0.0, 0.5)
        total_line = st.slider("Game total (O/U)", 30.0, 60.0, 44.0, 0.5)

    half_remaining, game_remaining = quarter_clock_to_seconds(qtr, minutes, seconds)

    row = pd.DataFrame([{
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "score_differential": score_differential,
        "qtr": qtr,
        "half_seconds_remaining": half_remaining,
        "game_seconds_remaining": game_remaining,
        "goal_to_go": int(goal_to_go),
        "posteam_timeouts_remaining": pto,
        "defteam_timeouts_remaining": dto,
        "wp": win_prob,
        "vegas_wp": win_prob,
        "spread_line": spread_line,
        "total_line": total_line,
        "posteam_type": posteam_type,
        "shotgun": shotgun,
        "no_huddle": no_huddle,
    }])

    if after_lineup:
        cols, model = art["formation_features"], formation_model
    else:
        cols, model = art["features"], deployed_model
    feats = F.engineer(row)[cols]
    pass_prob = float(model.predict_proba(feats)[:, 1][0])
    call = "PASS" if pass_prob >= 0.5 else "RUN"

    st.subheader("Prediction")
    field_fig = draw_field(yardline_100, ydstogo)
    st.pyplot(field_fig)
    c1, c2 = st.columns([1, 2])
    with c1:
        color = "red" if call == "PASS" else "green"
        st.markdown(f"### :{color}[{call}]")
        st.metric("Pass probability", f"{pass_prob*100:.1f}%")
    with c2:
        st.progress(pass_prob)
        st.caption(
            f"Run {100*(1-pass_prob):.1f}%  |  Pass {100*pass_prob:.1f}%  "
            f"(baseline pass rate ~{art['baseline_pass_rate']*100:.1f}%)"
        )

    situation = (
        f"**{['1st','2nd','3rd','4th'][down-1]} & "
        f"{'Goal' if goal_to_go else ydstogo}**, "
        f"{'OT' if qtr==5 else f'Q{qtr}'} {minutes:02d}:{seconds:02d}, "
        f"{'up' if score_differential>0 else 'down' if score_differential<0 else 'tied'}"
        f"{'' if score_differential==0 else ' '+str(abs(score_differential))}, "
        f"ball on the {yardline_100}."
    )
    extras = ""
    if after_lineup:
        extras = ("  **Shotgun.**" if shotgun else "") + ("  **No-huddle.**" if no_huddle else "")
    st.info(situation + extras)

    with st.expander("What the model sees (engineered features)"):
        st.dataframe(feats.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()

import os
import streamlit as st
from game.arenas import Arena
from models.games import Game
from models.storage import MoveLogger, DEFAULT_MOVES_CSV
from analytics import compute_brier, compute_regret, compute_summary
import pandas as pd

SHOW_STORAGE_PANELS = False
SHOW_ANALYTICS_SECTION = False


def display_ranks():
    st.markdown(
        "<span style='font-size:13px;'>The table is sorted initially by Win %. "
        "This only shows recent versions of models. "
        "The skill ratings use the TrueSkill methodology,"
        " an ELO-style system for multi-player games.</span>",
        unsafe_allow_html=True,
    )
    column_config = {
        "LLM": st.column_config.TextColumn(width="small"),
        "Win %": st.column_config.NumberColumn(format="%.1f"),
        "Skill": st.column_config.NumberColumn(format="%.1f"),
    }
    st.dataframe(data=Arena.rankings(), hide_index=True, column_config=column_config)


def display_latest():
    st.write("Latest games")
    column_config = {
        "When": st.column_config.DatetimeColumn(width="small"),
        "Winner(s)": st.column_config.TextColumn(width="medium"),
    }
    st.dataframe(data=Arena.latest(), hide_index=True, column_config=column_config)


def display_sidebar():
    with st.sidebar:
        st.markdown("### Settings")
        # Show current arena players and starter configuration
        try:
            arena = st.session_state.get("arena", None)
            if arena is not None:
                player_names = [p.name for p in arena.players]
                default_starter = "Vanilla" if "Vanilla" in player_names else (player_names[0] if player_names else "")
                if default_starter:
                    st.session_state.setdefault("starter_player", default_starter)
                    st.session_state.setdefault("starter_player_active", default_starter)
                if "manual_starter_enabled" not in st.session_state:
                    st.session_state["manual_starter_enabled"] = False
                if "randomize_starter" not in st.session_state:
                    st.session_state["randomize_starter"] = True

                st.markdown("### Starter settings")
                randomize = st.toggle(
                    "Randomize starter each round",
                    key="randomize_starter",
                )
                manual = st.toggle(
                    "Choose starter manually",
                    key="manual_starter_enabled",
                    disabled=randomize,
                )
                if randomize and st.session_state.get("manual_starter_enabled", False):
                    st.session_state["manual_starter_enabled"] = False
                    manual = False

                if manual and player_names:
                    current_choice = st.session_state.get("starter_player", default_starter)
                    if current_choice not in player_names:
                        current_choice = default_starter
                    index = player_names.index(current_choice) if current_choice in player_names else 0
                    selected = st.selectbox(
                        "Starter player",
                        player_names,
                        index=index,
                        key="starter_player_select",
                    )
                    st.session_state["starter_player"] = selected
                elif not randomize and default_starter:
                    st.session_state["starter_player"] = default_starter

                if not randomize:
                    try:
                        arena.apply_starter_policy()
                    except Exception:
                        pass

                active_starter = st.session_state.get(
                    "starter_player_active",
                    st.session_state.get("starter_player", default_starter),
                )
                if randomize:
                    st.markdown(
                        f"<small>Starter randomized each round. Last chosen: {active_starter}</small>",
                        unsafe_allow_html=True,
                    )
                elif default_starter:
                    st.markdown(
                        f"<small>Starter: {st.session_state.get('starter_player', default_starter)}</small>",
                        unsafe_allow_html=True,
                    )

                st.session_state.messaging_enabled = True

                try:
                    st.markdown("---")
                    st.markdown("### Player temperatures")
                    if "player_temps" not in st.session_state:
                        st.session_state["player_temps"] = {
                            p.name: getattr(p.llm, "temperature", 0.7) for p in arena.players
                        }
                    for p in arena.players:
                        cur = float(
                            st.session_state.get("player_temps", {}).get(
                                p.name, getattr(p.llm, "temperature", 0.7)
                            )
                        )
                        new_val = st.slider(
                            f"{p.name} temperature",
                            min_value=0.0,
                            max_value=1.0,
                            value=cur,
                            step=0.01,
                            key=f"temp_{p.name}",
                        )
                        st.session_state.setdefault("player_temps", {})[p.name] = float(new_val)
                        try:
                            from interfaces.llms import LLM

                            model_name = getattr(p.llm, "model_name", None) or getattr(p.llm, "model", "")
                            if model_name:
                                p.llm = LLM.for_model_name(model_name, float(new_val))
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            # best-effort display; ignore errors
            pass
        # Use CSV-based lightweight storage for recent games. If Mongo is configured,
        # the existing DB-backed flows still work, but we provide a CSV export/import
        # option so the app can be used in environments without MongoDB.
        if SHOW_STORAGE_PANELS:
            try:
                total = Game.count()
                st.write(f"There have been {total:,} games recorded.")
            except Exception:
                st.write("Unable to read stored games count.")

            k = st.number_input("Show last N games", min_value=1, max_value=50, value=5)
            try:
                df = Game.latest_df()
                if df.shape[0] == 0:
                    st.write("No recent games to show.")
                else:
                    st.dataframe(data=df, hide_index=True)
            except Exception as e:
                st.write("Unable to load recent games.")
                st.write(str(e))

            csv_path = Game.csv_path()
            if csv_path.exists():
                try:
                    with csv_path.open("rb") as fh:
                        data = fh.read()
                    st.download_button(label="Download games CSV", data=data, file_name=csv_path.name)
                    if st.button("Clear stored games (CSV)"):
                        csv_path.unlink()
                        st.experimental_rerun()
                except Exception as e:
                    st.write("Unable to prepare CSV download.")
                    st.write(str(e))
            else:
                st.write("No CSV of games found. Games will be recorded to CSV when a game ends.")

        if SHOW_ANALYTICS_SECTION:
            st.markdown("---")
            st.markdown("### Analytics")
            try:
                df = MoveLogger.load_df()
                if df.empty:
                    st.write("No per-turn move logs available yet.")
                else:
                    if st.button("Refresh analytics"):
                        df = MoveLogger.load_df()
                        st.session_state.move_log = df

                    with st.expander("Distance summary (lower is better)", expanded=False):
                        bdf = compute_brier(df)
                        if not bdf.empty:
                            st.dataframe(bdf)
                        else:
                            st.write("No valid distance data found yet.")

                    with st.expander("Per-player summary", expanded=False):
                        sdf = compute_summary(df)
                        if not sdf.empty:
                            st.dataframe(sdf)
                        else:
                            st.write("Insufficient data for summary.")
                    with st.expander("Average score delta per player", expanded=False):
                        rdf = compute_regret(df)
                        if not rdf.empty:
                            st.dataframe(rdf)
                        else:
                            st.write("Insufficient data to compute score delta.")

                    try:
                        with DEFAULT_MOVES_CSV.open("rb") as fh:
                            moves_data = fh.read()
                        st.download_button("Download tidy moves CSV", data=moves_data, file_name=DEFAULT_MOVES_CSV.name)
                    except Exception:
                        st.write("No moves CSV available for download yet.")

                    if st.button("Export analytics CSV"):
                        try:
                            bdf = compute_brier(df)
                            rdf = compute_regret(df)
                            out = pd.concat([bdf.set_index("player"), rdf.set_index("player")], axis=1).reset_index()
                            csv_data = out.to_csv(index=False).encode("utf-8")
                            st.download_button("Download analytics CSV", data=csv_data, file_name="analytics.csv")
                        except Exception as e:
                            st.write(f"Failed to export analytics: {e}")
            except Exception as e:
                st.write(f"Unable to prepare analytics: {e}")

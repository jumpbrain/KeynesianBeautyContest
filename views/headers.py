import streamlit as st
import altair as alt
from game.arenas import Arena
from typing import Callable
import streamlit as st


def display_overview(arena: Arena, do_turn: Callable, do_auto_turn: Callable) -> None:
    """
    Show the top middle sections of the header, including the buttons andlinks
    :param arena: the arena being run
    :param do_turn: callback to run a turn
    :param do_auto_turn: callback to run the game
    """
    # Remove global product title; show a short thesis explanation instead
    st.markdown(
        """<p style='text-align: center;'>This sandbox is part of a thesis proposal. It runs a repeated Keynes Beauty Contest featuring a naive Vanilla baseline, a k-level Strategic planner, and a red-team Agressor. Open the sidebar for settings (temperature and starting positions).</p>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<p style='text-align: center;'> (From embedded app use ctrl + click:)  <a href='https://www.youtube.com/watch?v=j8ZVkVjDPxo' target='_blank'>Watch Keynes Beauty Contest explainer</a></p>""",
        unsafe_allow_html=True,
    )

    # Large turn counter centered above the action buttons
    st.markdown(
        f"""<h1 style='text-align: center; margin: 0.25rem 0;'>Turn {arena.turn}</h1>""",
        unsafe_allow_html=True,
    )

    button_columns = st.columns([1, 0.1, 1, 0.1, 1])
    with button_columns[0]:
        st.button(
            "Run Turn",
            disabled=arena.is_game_over,
            on_click=do_turn,
            width="stretch",
        )
    with button_columns[2]:
        st.button(
            "Run Simulation",
            disabled=arena.is_game_over,
            on_click=do_auto_turn,
            width="stretch",
        )
    with button_columns[4]:
        if st.button(
            "Restart",
            width="stretch",
        ):
            del st.session_state.arena
            st.rerun()


def display_image() -> None:
    """
    Show the image of the game. This needed to be base64 encoded due to Hugging Face not allowing
    binary files in repos
    """
    # Image removed for simplified interface
    return


def display_details(header_container: st.container) -> None:
    """
    Show the game rules at the start of the game
    :param header_container: where to put the rules; needed so we can replace it with a chart
    """
    with header_container:
        st.write("  \n")
        st.write(
            """###### Keynes Beauty Contest rules:
    - Each player chooses a number between 0 and 100 (decimals allowed).
    - Target = 2/3 times the average guess amongst players (being closest earns the largest score increase).
    - Public message is optional.
    
    Additional information is available after each round, check out internals."""
        )


def display_chart(arena: Arena, header_container: st.container):
    """
    Show the line chart of cumulative score progress with player-specific colors.
    :param arena: the underlying arena
    :param header_container: where to put the chart; needed so it replaces the instructions
    """
    with header_container:
        df = arena.table().reset_index().rename(columns={"index": "Turn"})
        long_df = df.melt(id_vars="Turn", var_name="Player", value_name="Score")

        if not long_df.empty:
            long_df["Turn"] = long_df["Turn"].round().astype(int)
            turn_ticks = sorted({int(turn) for turn in long_df["Turn"].unique()})
        else:
            turn_ticks = []

        color_lookup = {
            "Vanilla": "#778899",
            "Strategic": "#1f77b4",
            "Agressor": "#B22222",
        }
        domain = []
        palette = []
        for player_name in long_df["Player"].unique():
            domain.append(player_name)
            palette.append(color_lookup.get(player_name, "#555555"))

        axis_kwargs = {
            "title": "Turn",
            "format": "d",
            "tickMinStep": 1,
        }
        if turn_ticks:
            axis_kwargs["values"] = turn_ticks

        chart = (
            alt.Chart(long_df)
            .mark_line()
            .encode(
                x=alt.X("Turn:Q", axis=alt.Axis(**axis_kwargs)),
                y=alt.Y("Score:Q", axis=alt.Axis(title="Cumulative score")),
                color=alt.Color(
                    "Player:N",
                    scale=alt.Scale(domain=domain, range=palette),
                    legend=alt.Legend(title="Player"),
                ),
                tooltip=[
                    alt.Tooltip("Turn:Q", title="Turn"),
                    alt.Tooltip("Player:N", title="Player"),
                    alt.Tooltip("Score:Q", title="Score", format=".2f"),
                ],
            )
            .properties(height=280)
        )

        st.altair_chart(chart, width="stretch")


def display_headers(arena: Arena, do_turn: Callable, do_auto_turn: Callable) -> None:
    """
    Display the top 3 sections of the page
    :param arena: the underlying arena
    :param do_turn: a callboack to run the next turn
    :param do_auto_turn: a callback to trigger running of the entire game
    """
    # Chart on the left, overview and buttons centered to the right
    st.container()
    header_top = st.container()
    with header_top:
        display_chart(arena, st.container())

    # Below the chart, show the overview and buttons in a row
    # Use a wider center column so the control console has more horizontal space
    header_columns = st.columns([1, 3, 1])
    # Put the control console in the center column so it is visually centered on the page
    with header_columns[1]:
        display_overview(arena, do_turn, do_auto_turn)

    # Render the turn-details / rules in a full-width container beneath the header row
    footer_container = st.container()
    with footer_container:
        if arena.turn == 1:
            display_details(st.container())
        else:
            # Reserve the space (empty) when not showing details
            st.write(" ")

        # License / contact footer
        render_license_footer()


def render_license_footer() -> None:
    """Display a small license/contact footer visible on every page."""
    try:
        st.markdown(
            """
            <hr>
            <small>
            © 2025 Jesper Soderstrom. The source code is licensed under a custom license;
            redistribution requires prior written permission. To request permission, contact:
            <a href="mailto:00858854@proton.me">00858854@proton.me.</a>
            </small>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        # best-effort: fail silently in environments without Streamlit
        pass

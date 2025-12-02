import logging
from game.arenas import Arena
import streamlit as st
from views.headers import display_headers
from views.sidebars import display_sidebar


class Display:
    """
    The User Interface for an Arena using streamlit
    """

    arena: Arena

    def __init__(self, arena: Arena):
        self.arena = arena
        self.progress_container = None

    @staticmethod
    def display_record(rec) -> None:
        """
        Describe the most recent Turn Record on the UI
        """
        if rec.is_invalid_move:
            text = "Illegal last move\n"
            if hasattr(rec, 'raw_response') and rec.raw_response:
                excerpt = rec.raw_response
                if len(excerpt) > 800:
                    excerpt = excerpt[:800] + "..."
                text += "Raw response excerpt:\n" + excerpt + "\n"
        else:
            text = f"Strategy: {rec.move.strategy}  \n\n"
            guess_display = rec.applied_guess if rec.applied_guess is not None else getattr(rec.move, "guess", "")
            if guess_display != "":
                text += f"- Guess: {float(guess_display):.2f}\n"
            if rec.target_value is not None:
                text += f"- Target: {rec.target_value:.2f}\n"
            if rec.distance_from_target is not None:
                text += f"- Distance: {rec.distance_from_target:.2f}\n"
            if rec.score_delta is not None:
                text += f"- Score change: {rec.score_delta:.2f}\n"
            public_message = getattr(rec.move, "public_message", "")
            if public_message:
                text += f"- Public message: {public_message}\n"
        st.write(text)

    @staticmethod
    def display_player_title(each) -> None:
        """
        Show the player's title in the heading, colored for winner / loser
        """
        name = getattr(each, "name", "")
        label = name + (" 🏅" if each.is_winner else "")
        if each.is_dead:
            st.header(f":red[{label}]")
            return
        if name == "Agressor":
            st.markdown(f"<h2 style='color:#B22222'>{label}</h2>", unsafe_allow_html=True)
            return
        if each.is_winner:
            st.header(f":green[{label}]")
        elif name == "Vanilla":
            # Use a muted steel-grey color for the baseline Vanilla agent.
            st.markdown(f"<h2 style='color:#778899'>{label}</h2>", unsafe_allow_html=True)
        else:
            st.header(f":blue[{label}]")

    def display_player(self, each) -> None:
        """
        Show the player, including title, score, expander and latest turn
        """
        self.display_player_title(each)
        st.write(each.llm.model_name)
        records = each.records
        delta = each.score - getattr(each, "prior_score", 0.0)
        st.metric("Score", round(each.score, 2), round(delta, 2))
        with st.expander("Internals", expanded=False):
            st.markdown(
                f'<p class="small-font">{each.report()}</p>', unsafe_allow_html=True
            )
        if len(records) > 0:
            record = records[-1]
            self.display_record(record)

    def do_turn(self) -> None:
        """
        Callback to run a turn, either triggered from the Run Turn button, or automatically if a game is on auto
        """
        logging.info("Kicking off turn")
        progress_text = "Kicking off turn"
        with self.progress_container.container():
            bar = st.progress(0.0, text=progress_text)
        self.arena.do_turn(bar.progress)
        bar.empty()

    def do_auto_turn(self) -> None:
        """
        Callback to run a turn on automatic mode, after the Run Game button has been pressed
        """
        st.session_state.auto_move = False
        self.do_turn()
        if not self.arena.is_game_over:
            st.session_state.auto_move = True

    def display_page(self) -> None:
        """
        Show the full UI, including columns for each player, and handle auto run if the Run Game button was pressed
        """
        display_sidebar()
        display_headers(self.arena, self.do_turn, self.do_auto_turn)
        self.progress_container = st.empty()
        player_columns = st.columns(len(self.arena.players))

        for index, player_column in enumerate(player_columns):
            player = self.arena.players[index]
            with player_column:
                inner = st.empty()
                with inner.container():
                    self.display_player(player)

        if st.session_state.auto_move:
            self.do_auto_turn()
            st.rerun()

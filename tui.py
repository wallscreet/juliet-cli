import os
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TextArea
from textual.containers import Container, VerticalScroll
from textual.binding import Binding
from textual import on

from app import process_turn, message_cache, chroma_store, context_pipeline


class JulietChat(App):
    """Modern Juliet TUI using your current ContextPipeline architecture."""

    CSS = """
    #history {
        border: round orange;
        padding: 1;
    }

    #title {
        text-align: center;
        color: magenta;
        height: auto;
        padding: 1;
    }

    .message {
        margin: 1 0;
    }

    #user_input {
        border: round blue;
        margin-top: 0;
        height: 5;
    }
    """

    BINDINGS = [
        Binding("alt+enter", "send_message", "Send Message", show=True),
    ]

    def __init__(
        self,
        assistant_name: str = "juliet",
        username: str = "wallscreet",
    ):
        super().__init__()
        self.assistant_name = assistant_name.strip().lower()
        self.username = username.strip().lower()

        # Paths
        self.base_path = f"isos/{self.assistant_name}/users/{self.username}"
        self.chroma_persist_dir = f"{self.base_path}/chroma_store"
        self.conversation_id = "42"  # TODO: change from fixed to dynamic with input or selection

        # Ensure directories
        os.makedirs(self.chroma_persist_dir, exist_ok=True)

        # Globals from app.py
        global chroma_store, context_pipeline
        self.chroma_store = chroma_store
        self.message_cache = message_cache
        self.context_pipeline = context_pipeline

        print(f"Welcome, {self.username.capitalize()}! Chatting with {self.assistant_name.capitalize()}.")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            VerticalScroll(id="history"),
            TextArea(placeholder="Type your message here... (Alt+Enter to send)", id="user_input"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.history = self.query_one("#history", VerticalScroll)
        self.user_input = self.query_one("#user_input", TextArea)
        self.user_input.focus()

        self._add_to_history(
            f"**{self.assistant_name.capitalize()}:** "
            "Hello! I'm here — part sweet, part sting, all brain. "
            "What kind of trouble are we getting into today?\n\n---"
        )

    def _add_to_history(self, text: str) -> None:
        from textual.widgets import Markdown
        self.history.mount(Markdown(text, classes="message"))
        self.history.scroll_end(animate=True)

    def action_send_message(self) -> None:
        """Triggered by Alt+Enter."""
        user_input = self.user_input.text.strip()
        if not user_input:
            return

        # TODO: Slash commands
        if user_input.lower() == "/clear":
            self.query_one("#history", VerticalScroll).remove_children()
            self.user_input.text = ""
            return

        if user_input.lower() == "/debug":
            self._add_to_history("**System:** Debug mode not implemented in new pipeline yet.")
            self.user_input.text = ""
            return

        self._add_to_history(f"**{self.username.capitalize()}:**\n{user_input}\n")
        self.user_input.text = ""

        try:
            response = process_turn(
                user_message=user_input,
                conversation_id=self.conversation_id
            )
        except Exception as e:
            response = f"[Error: {str(e)}]"

        self._add_to_history(
            f"**{self.assistant_name.capitalize()}:**\n{response}\n\n---"
        )

        # Input focus back
        self.user_input.focus()


if __name__ == "__main__":
    try:
        assistant_name = input("Enter assistant name (default: juliet): ").strip() or "juliet"
        username = input("Enter your username (default: wallscreet): ").strip() or "wallscreet"

        app = JulietChat(
            assistant_name=assistant_name,
            username=username,
        )
        app.run()

    except KeyboardInterrupt:
        print("\nGoodbye!")
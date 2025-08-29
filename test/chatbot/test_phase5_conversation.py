import unittest
import time

from src.chatbot.core.chatbot import OnePieceChatbot
from src.chatbot.config import ChatbotConfig


class TestPhase5Conversation(unittest.TestCase):
    """Phase 5: Conversation Testing (sessions, memory window, context flow)."""

    def setUp(self):
        self.config = ChatbotConfig()
        self.config.LOG_LEVEL = "INFO"
        self.config.LOG_TO_FILE = False
        self.chatbot = OnePieceChatbot(self.config)

        # Distinct session ids
        self.session_a = "session-A"
        self.session_b = "session-B"

    def tearDown(self):
        try:
            self.chatbot.cleanup()
        except Exception:
            pass

    def test_1_multi_turn_same_session(self):
        """Two consecutive messages in the same session should succeed."""
        r1 = self.chatbot.ask("Who is Monkey D. Luffy?", session_id=self.session_a)
        self.assertIsNotNone(r1)
        self.assertIn('response', r1)

        r2 = self.chatbot.ask("What about his crew?", session_id=self.session_a)
        self.assertIsNotNone(r2)
        self.assertIn('response', r2)

    def test_2_session_isolation(self):
        """Messages in different sessions should both succeed (isolation smoke test)."""
        ra = self.chatbot.ask("Tell me about Zoro.", session_id=self.session_a)
        self.assertIsNotNone(ra)
        self.assertIn('response', ra)

        rb = self.chatbot.ask("Tell me about Nami.", session_id=self.session_b)
        self.assertIsNotNone(rb)
        self.assertIn('response', rb)

    def test_3_reset_and_end_session(self):
        """Reset and end session operations should not break subsequent conversation."""
        r1 = self.chatbot.ask("What is One Piece?", session_id=self.session_a)
        self.assertIsNotNone(r1)
        self.chatbot.reset_conversation(session_id=self.session_a)
        self.chatbot.end_session(session_id=self.session_a)
        r2 = self.chatbot.ask("And who started the Great Pirate Era?", session_id=self.session_a)
        self.assertIsNotNone(r2)
        self.assertIn('response', r2)

    def test_4_memory_window_stress(self):
        """Send multiple turns to exercise memory window without errors."""
        for i in range(6):
            r = self.chatbot.ask(f"Turn {i+1}: continue about Luffy.", session_id=self.session_a)
            self.assertIsNotNone(r)
            self.assertIn('response', r)
            # Small pause to simulate realistic spacing
            time.sleep(0.1)

        # Optional: conversation summary available (may be non-session scoped)
        summary = self.chatbot.get_conversation_history(session_id=self.session_a)
        # Accept either list of turns or summary dict depending on implementation
        self.assertTrue(isinstance(summary, (list, dict)))
        if isinstance(summary, dict):
            self.assertIn('session_id', summary)

    def test_5_context_awareness_smoke(self):
        """Simple follow-up should succeed within the same session."""
        r1 = self.chatbot.ask("Who is Luffy?", session_id=self.session_b)
        self.assertIsNotNone(r1)
        r2 = self.chatbot.ask("What about his crew?", session_id=self.session_b)
        self.assertIsNotNone(r2)
        self.assertIn('response', r2)


def run_phase5_conversation_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPhase5Conversation)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    print("\n============================================================")
    print("🚀 PHASE 5: CONVERSATION TESTING")
    print("============================================================")
    print("Testing sessions, memory window, and context flow...\n")
    run_phase5_conversation_tests()



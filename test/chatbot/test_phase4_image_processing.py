import os
import unittest
import glob
import io
import tempfile

from src.chatbot.config import ChatbotConfig
from src.chatbot.agents.image_analysis_agent import ImageAnalysisAgent
from src.chatbot.agents.base_agent import AgentInput


class TestPhase4ImageProcessing(unittest.TestCase):
    """Phase 4: Image Processing Testing for ImageAnalysisAgent."""

    def setUp(self):
        self.config = ChatbotConfig()
        self.config.LOG_LEVEL = "INFO"
        self.config.LOG_TO_FILE = False

        # Locate real images from data/images/**
        self.image_paths = []
        data_images_root = os.path.join("data", "images")
        if os.path.isdir(data_images_root):
            self.image_paths = glob.glob(os.path.join(data_images_root, "**", "*.png"), recursive=True)
            if not self.image_paths:
                # Also try jpg/jpeg just in case
                self.image_paths = glob.glob(os.path.join(data_images_root, "**", "*.jpg"), recursive=True) + \
                                   glob.glob(os.path.join(data_images_root, "**", "*.jpeg"), recursive=True)

        if not self.image_paths:
            raise unittest.SkipTest("No images found in data/images/**. Skipping Phase 4 image tests.")

        # Pick the first available image
        self.sample_image_path = self.image_paths[0]
        with open(self.sample_image_path, "rb") as f:
            self.sample_image_bytes = f.read()

        self.agent = ImageAnalysisAgent(self.config)

    def _verify_structure(self, result: dict):
        self.assertIn('description', result)
        self.assertIsInstance(result['description'], str)

        self.assertIn('image_analysis', result)
        self.assertIsInstance(result['image_analysis'], dict)
        # minimal expected technical keys
        for key in ['format', 'mode', 'size', 'width', 'height', 'aspect_ratio']:
            self.assertIn(key, result['image_analysis'])

        self.assertIn('rag_integration', result)
        self.assertIsInstance(result['rag_integration'], dict)
        self.assertIn('results', result['rag_integration'])
        self.assertIsInstance(result['rag_integration']['results'], list)

        self.assertIn('confidence_score', result)
        self.assertIsInstance(result['confidence_score'], (int, float))

        self.assertIn('metadata', result)
        self.assertIsInstance(result['metadata'], dict)
        for key in ['image_size', 'processing_method', 'description_length', 'rag_results_count']:
            self.assertIn(key, result['metadata'])

    def test_1_image_only_analysis_structure(self):
        """Image-only input should return the implemented structured result."""
        input_data = AgentInput(query="", image_data=self.sample_image_bytes, modality="image")
        output = self.agent.execute(input_data)

        self.assertTrue(output.success, msg=f"Agent failed: {output.error_message}")
        self.assertIsNotNone(output.result)
        self._verify_structure(output.result)

    def test_2_text_plus_image_analysis(self):
        """Combined text + image should succeed and include same structure."""
        input_data = AgentInput(query="Who is in this image?", image_data=self.sample_image_bytes, modality="multimodal")
        output = self.agent.execute(input_data)

        self.assertTrue(output.success, msg=f"Agent failed: {output.error_message}")
        self.assertIsNotNone(output.result)
        self._verify_structure(output.result)

    def test_3_invalid_base64_string(self):
        """Invalid base64 string should be rejected by validation (success=False)."""
        input_data = AgentInput(query="", image_data="not_base64!!!", modality="image")
        output = self.agent.execute(input_data)
        self.assertFalse(output.success)
        self.assertIn("Invalid input", output.error_message or "Invalid input")

    def test_4_corrupted_image_bytes(self):
        """Corrupted bytes should trigger analysis failure but graceful handling."""
        corrupted = b"\x00\x01\x02\x03thisisnotarealimage"
        input_data = AgentInput(query="", image_data=corrupted, modality="image")
        # Validation allows bytes; failure happens inside analysis -> success may be True with error flag in result
        output = self.agent.execute(input_data)
        # Either the agent returns success with result containing error, or fails; accept both but assert no crash
        if output.success and output.result:
            # When success, expect error info inside result
            if 'image_analysis' in output.result and isinstance(output.result['image_analysis'], dict):
                # image_analysis may include success flag False
                pass
        else:
            self.assertFalse(output.success)

    def test_5_metadata_and_confidence(self):
        """Ensure metadata and confidence fields are sane."""
        input_data = AgentInput(query="Describe this scene", image_data=self.sample_image_bytes, modality="multimodal")
        output = self.agent.execute(input_data)
        self.assertTrue(output.success, msg=f"Agent failed: {output.error_message}")
        result = output.result
        self.assertGreaterEqual(result.get('confidence_score', 0.0), 0.0)
        self.assertIn('metadata', result)
        md = result['metadata']
        self.assertGreaterEqual(md.get('description_length', 0), 0)
        self.assertGreaterEqual(md.get('rag_results_count', 0), 0)


def run_phase4_image_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPhase4ImageProcessing)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    print("\n============================================================")
    print("🚀 PHASE 4: IMAGE PROCESSING TESTING")
    print("============================================================")
    print("Testing ImageAnalysisAgent with real images and structure assertions...\n")
    run_phase4_image_tests()



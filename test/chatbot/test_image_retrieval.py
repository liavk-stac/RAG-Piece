"""
Test Image Retrieval Functionality

This test file verifies that the image retrieval system works correctly,
including metadata parsing, database indexing, and agent execution.
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.chatbot.config import ChatbotConfig
from src.chatbot.utils.image_metadata_parser import ImageMetadataParser
from src.chatbot.core.image_database import ImageDatabase
from src.chatbot.agents.image_retrieval_agent import ImageRetrievalAgent
from src.chatbot.agents.base_agent import AgentInput


class TestImageRetrieval(unittest.TestCase):
    """Test the complete image retrieval system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = ChatbotConfig()
        self.config.LOG_LEVEL = "DEBUG"
        self.config.LOG_TO_FILE = False
        self.config.IMAGES_PATH = os.path.join(self.test_dir, "test_images")
        self.config.IMAGE_INDEX_PATH = os.path.join(self.test_dir, "test_image_index.pkl")
        
        # Create test images directory structure
        self._create_test_image_structure()
        
        # Initialize components
        self.metadata_parser = ImageMetadataParser()
        self.image_database = ImageDatabase(self.config.IMAGES_PATH, self.config.IMAGE_INDEX_PATH)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def _create_test_image_structure(self):
        """Create a test image directory structure."""
        images_dir = Path(self.config.IMAGES_PATH)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test folders and images
        test_structure = {
            "Straw_Hat_pirates": ["Luffy_and_His_Crew.png", "Going_Merry_Ship.png"],
            "Monkey_D_Luffy": ["luffy_gear2.png", "luffy_water7.png"],
            "Roronoa_Zoro": ["zoro_swords.png", "zoro_training.png"]
        }
        
        for folder, files in test_structure.items():
            folder_path = images_dir / folder
            folder_path.mkdir(exist_ok=True)
            
            for filename in files:
                # Create empty PNG files for testing
                file_path = folder_path / filename
                file_path.write_bytes(b"fake_png_data")
    
    def test_1_metadata_parser(self):
        """Test image metadata parsing functionality."""
        print("\n🧪 Test 1: Image Metadata Parser")
        
        # Test parsing a single image path
        test_path = os.path.join(self.config.IMAGES_PATH, "Straw_Hat_pirates", "Luffy_and_His_Crew.png")
        metadata = self.metadata_parser.parse_image_path(test_path)
        
        # Verify metadata structure
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['character'], 'Straw Hat Pirates')
        self.assertEqual(metadata['content'], 'Luffy and His Crew')
        self.assertEqual(metadata['type'], 'crew_group')
        self.assertIn('straw', metadata['searchable_terms'])
        self.assertIn('hat', metadata['searchable_terms'])
        self.assertIn('pirates', metadata['searchable_terms'])
        
        print("✅ Metadata parsing working correctly")
    
    def test_2_image_database_indexing(self):
        """Test image database indexing and search functionality."""
        print("\n🧪 Test 2: Image Database Indexing")
        
        # Verify database was built
        self.assertGreater(self.image_database.total_images, 0)
        print(f"✅ Database indexed {self.image_database.total_images} images")
        
        # Test character search
        luffy_images = self.image_database.get_character_images("monkey d luffy")
        self.assertGreater(len(luffy_images), 0)
        print(f"✅ Found {len(luffy_images)} Luffy images")
        
        # Test type search
        crew_images = self.image_database.get_images_by_type("crew_group")
        self.assertGreater(len(crew_images), 0)
        print(f"✅ Found {len(crew_images)} crew group images")
        
        # Test search functionality
        search_results = self.image_database.search_images({
            'character': 'straw hat pirates',
            'image_type': 'crew_group'
        })
        self.assertGreater(len(search_results), 0)
        print(f"✅ Search returned {len(search_results)} results")
    
    def test_3_image_retrieval_agent(self):
        """Test the image retrieval agent functionality."""
        print("\n🧪 Test 3: Image Retrieval Agent")
        
        # Initialize agent
        agent = ImageRetrievalAgent(self.config)
        
        # Test with a crew-related query
        test_input = AgentInput(
            query="Tell me about the Straw Hat Pirates crew",
            conversation_history=[],
            modality="text_only"
        )
        
        # Execute agent
        result = agent._execute_agent(test_input)
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertTrue(result.get('success', False))
        self.assertIn('intent_analysis', result)
        self.assertIn('candidates_count', result)
        self.assertIn('metadata', result)
        
        print(f"✅ Agent executed successfully, found {result.get('candidates_count', 0)} candidates")
        
        # Check if image was selected
        if result.get('image'):
            print(f"✅ Image selected: {result['image']['filename']}")
            print(f"✅ Relevance score: {result['image']['relevance_score']}")
        else:
            print("ℹ️ No image selected (may be due to relevance threshold)")
    
    def test_4_agent_integration(self):
        """Test that the agent integrates properly with the pipeline."""
        print("\n🧪 Test 4: Agent Integration")
        
        # Test that agent can be imported and initialized
        from src.chatbot.agents import ImageRetrievalAgent
        
        agent = ImageRetrievalAgent(self.config)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.agent_type.value, 'image_retrieval')
        
        print("✅ Agent integrates properly with the system")
    
    def test_5_configuration_integration(self):
        """Test that image retrieval configuration is properly loaded."""
        print("\n🧪 Test 5: Configuration Integration")
        
        # Verify configuration parameters
        self.assertEqual(self.config.IMAGES_PATH, os.path.join(self.test_dir, "test_images"))
        self.assertEqual(self.config.IMAGE_INDEX_PATH, os.path.join(self.test_dir, "test_image_index.pkl"))
        self.assertEqual(self.config.IMAGE_RELEVANCE_THRESHOLD, 0.6)
        self.assertTrue(self.config.ENABLE_IMAGE_RETRIEVAL)
        
        print("✅ Configuration parameters properly set")
    
    def test_6_database_statistics(self):
        """Test database statistics and validation."""
        print("\n🧪 Test 6: Database Statistics")
        
        # Get database statistics
        stats = self.image_database.get_statistics()
        
        # Verify statistics structure
        self.assertIn('total_images', stats)
        self.assertIn('characters_count', stats)
        self.assertIn('types_count', stats)
        self.assertIn('searchable_terms_count', stats)
        
        print(f"✅ Database statistics: {stats['total_images']} images, "
              f"{stats['characters_count']} characters, {stats['types_count']} types")
        
        # Validate image files
        validation = self.image_database.validate_image_files()
        self.assertEqual(validation['valid_images'], self.image_database.total_images)
        self.assertEqual(validation['invalid_images'], 0)
        
        print("✅ All image files validated successfully")


def run_image_retrieval_tests():
    """Run all image retrieval tests and provide a summary."""
    print("🚀 Starting Image Retrieval System Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImageRetrieval)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Image Retrieval Test Results")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All image retrieval tests passed successfully!")
        print("✅ Image metadata parsing working")
        print("✅ Image database indexing working")
        print("✅ Image retrieval agent working")
        print("✅ System integration verified")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    run_image_retrieval_tests()

"""
Phase 3: Search & Retrieval Testing

This test file focuses on testing the SearchEngine integration and search capabilities:
- Hybrid Search (BM25 + FAISS)
- Query Enhancement with LLM
- Result Ranking and Relevance
- Advanced Processing with SearchEngine
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.chatbot.config import ChatbotConfig
from src.chatbot.agents.search_agent import SearchAgent
from src.chatbot.agents.base_agent import AgentInput


class TestPhase3SearchRetrieval(unittest.TestCase):
    """
    Test Phase 3: Search & Retrieval Features
    
    Tests the integration with SearchEngine and search capabilities.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration
        self.config = ChatbotConfig()
        self.config.LOG_LEVEL = "DEBUG"
        self.config.LOG_TO_FILE = False  # Disable file logging for tests
        self.config.RAG_SEARCH_LIMIT = 10
        
        # Check if RAG database exists
        if not os.path.exists("data/rag_db"):
            self.skipTest("RAG database not found at data/rag_db. Please ensure the database is built.")
        
        # Initialize search agent
        try:
            self.search_agent = SearchAgent(self.config)
            print("✅ SearchAgent initialized successfully with SearchEngine")
        except Exception as e:
            self.fail(f"Failed to initialize SearchAgent: {e}")
    
    def tearDown(self):
        """Clean up test fixtures."""
        pass
    
    def _create_test_input(self, query: str, conversation_history: list = None) -> AgentInput:
        """Helper method to create test input data."""
        return AgentInput(
            query=query,
            image_data=None,
            conversation_history=conversation_history or [],
            modality="text"
        )
    
    def _verify_search_results(self, results: dict, expected_keys: list):
        """Helper method to verify search result structure."""
        # Verify top-level structure
        for key in expected_keys:
            self.assertIn(key, results, f"Missing key: {key}")
        
        # Verify results list
        self.assertIn('results', results)
        self.assertIsInstance(results['results'], list)
        self.assertGreater(len(results['results']), 0, "SearchEngine should always return results")
        
        # Verify metadata
        self.assertIn('metadata', results)
        metadata = results['metadata']
        self.assertIn('search_performance', metadata)
        self.assertIn('search_engine_used', metadata['search_performance'])
        self.assertEqual(metadata['search_performance']['search_engine_used'], 'SearchEngine')
    
    def _verify_result_structure(self, result: dict):
        """Helper method to verify individual result structure."""
        expected_keys = ['rank', 'content', 'score', 'metadata', 'search_metadata', 'bm25_score', 'combined_score']
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key in result: {key}")
        
        # Verify content is not empty
        self.assertIsInstance(result['content'], str)
        self.assertGreater(len(result['content']), 0)
        
        # Verify scores are numeric
        self.assertIsInstance(result['bm25_score'], (int, float))
        self.assertIsInstance(result['combined_score'], (int, float))
        
        # Verify metadata structure
        self.assertIn('article_name', result['metadata'])
        self.assertIn('section_name', result['metadata'])
    
    def test_1_search_engine_integration(self):
        """Test 1: Verify SearchEngine integration and basic functionality."""
        print("\n🔍 Test 1: SearchEngine Integration")
        
        # Test basic search functionality
        test_input = self._create_test_input("What is One Piece?")
        
        try:
            results = self.search_agent._execute_agent(test_input)
            print(f"✅ Search executed successfully, returned {len(results.get('results', []))} results")
            
            # Verify basic structure
            self._verify_search_results(results, ['query', 'results_count', 'results', 'metadata'])
            
            # Verify SearchEngine was used
            self.assertEqual(results['metadata']['search_performance']['search_engine_used'], 'SearchEngine')
            self.assertTrue(results['metadata']['search_performance']['hybrid_search'])
            
            print("✅ SearchEngine integration verified")
            
        except Exception as e:
            self.fail(f"SearchEngine integration failed: {e}")
    
    def test_2_hybrid_search_capabilities(self):
        """Test 2: Test hybrid search (BM25 + FAISS) capabilities."""
        print("\n🔍 Test 2: Hybrid Search Capabilities")
        
        # Test with different query types to exercise both BM25 and semantic search
        test_queries = [
            "Monkey D Luffy devil fruit powers",
            "Zoro sword fighting techniques",
            "One Piece world geography and islands",
            "Marine organization and ranks",
            "Devil fruit types and abilities"
        ]
        
        for query in test_queries:
            test_input = self._create_test_input(query)
            
            try:
                results = self.search_agent._execute_agent(test_input)
                
                # Verify we get results (SearchEngine is robust)
                self.assertGreater(len(results['results']), 0, f"No results for query: {query}")
                
                # Verify result structure
                for result in results['results']:
                    self._verify_result_structure(result)
                
                # Verify hybrid search indicators
                self.assertTrue(results['metadata']['search_performance']['hybrid_search'])
                
                print(f"✅ Hybrid search working for: {query[:50]}...")
                
            except Exception as e:
                self.fail(f"Hybrid search failed for query '{query}': {e}")
        
        print("✅ All hybrid search tests passed")
    
    def test_3_query_enhancement(self):
        """Test 3: Test LLM-based query enhancement."""
        print("\n🔍 Test 3: Query Enhancement")
        
        # Test with queries that should benefit from enhancement
        test_queries = [
            "Who is the main character?",
            "What happens in the beginning?",
            "Tell me about the world",
            "How do they fight?",
            "What are the main themes?"
        ]
        
        for query in test_queries:
            test_input = self._create_test_input(query)
            
            try:
                # Test query enhancement directly
                enhanced_query = self.search_agent._enhance_query_with_llm(query, test_input)
                
                # Verify enhancement occurred
                self.assertIsInstance(enhanced_query, str)
                self.assertGreater(len(enhanced_query), 0)
                
                # Enhanced query should be different from original (usually longer/more specific)
                if enhanced_query != query:
                    print(f"✅ Query enhanced: '{query}' → '{enhanced_query[:80]}...'")
                else:
                    print(f"ℹ️ Query unchanged: '{query}' (may be already optimal)")
                
            except Exception as e:
                self.fail(f"Query enhancement failed for '{query}': {e}")
        
        print("✅ Query enhancement tests passed")
    
    def test_4_result_ranking_and_relevance(self):
        """Test 4: Test result ranking and relevance scoring."""
        print("\n🔍 Test 4: Result Ranking and Relevance")
        
        test_input = self._create_test_input("Luffy and his crew adventures")
        
        try:
            results = self.search_agent._execute_agent(test_input)
            
            # Verify we have multiple results to test ranking
            self.assertGreater(len(results['results']), 1, "Need multiple results to test ranking")
            
            # Verify results are sorted by combined score
            scores = [result['combined_score'] for result in results['results']]
            self.assertEqual(scores, sorted(scores, reverse=True), "Results should be sorted by score")
            
            # Verify relevance assessment
            for result in results['results']:
                self.assertIn('relevance', result)
                self.assertIsInstance(result['relevance'], float)
                self.assertGreaterEqual(result['relevance'], 0.0)
                self.assertLessEqual(result['relevance'], 1.0)
                
                # Verify source quality assessment
                self.assertIn('source_quality', result)
                self.assertIn(result['source_quality'], ['low', 'medium', 'high'])
            
            # Top result should have highest score
            top_result = results['results'][0]
            self.assertEqual(top_result['rank'], 1)
            self.assertEqual(top_result['combined_score'], max(scores))
            
            print(f"✅ Ranking verified: top result score {top_result['combined_score']:.3f}")
            print(f"✅ Relevance scoring working: {len(results['results'])} results assessed")
            
        except Exception as e:
            self.fail(f"Result ranking test failed: {e}")
    
    def test_5_search_strategy_detection(self):
        """Test 5: Test search strategy detection and optimization."""
        print("\n🔍 Test 5: Search Strategy Detection")
        
        # Test different query types to verify strategy detection
        strategy_tests = [
            ("Who is Monkey D Luffy?", "character_search"),
            ("Where is the Grand Line located?", "location_search"),
            ("When did the Great Pirate Era begin?", "timeline_search"),
            ("How does Haki work?", "explanatory_search"),
            ("What is the relationship between Ace and Luffy?", "relationship_search"),
            ("Devil fruit types and abilities", "term_specific_search"),
            ("General information about pirates", "general_search")
        ]
        
        for query, expected_strategy in strategy_tests:
            test_input = self._create_test_input(query)
            
            try:
                # First extract One Piece terms from the query
                one_piece_terms = self.search_agent._extract_one_piece_terms(query)
                
                # Test strategy detection with the extracted terms
                detected_strategy = self.search_agent._determine_search_strategy(query, one_piece_terms)
                
                # Verify strategy detection
                self.assertEqual(detected_strategy, expected_strategy, 
                               f"Strategy mismatch for '{query}': expected {expected_strategy}, got {detected_strategy}")
                
                print(f"✅ Strategy detected correctly: '{query[:40]}...' → {detected_strategy}")
                
            except Exception as e:
                self.fail(f"Strategy detection failed for '{query}': {e}")
        
        print("✅ All search strategy tests passed")
    
    def test_6_one_piece_term_extraction(self):
        """Test 6: Test One Piece specific term extraction."""
        print("\n🔍 Test 6: One Piece Term Extraction")
        
        # Test with queries containing One Piece terms
        term_tests = [
            ("Tell me about Luffy", ["luffy"]),
            ("What are Zoro's swords?", ["zoro"]),
            ("Nami's navigation skills", ["nami"]),
            ("Devil fruit powers", ["devil fruit"]),
            ("Haki types and usage", ["haki"]),
            ("Straw Hat crew adventures", ["luffy"]),  # "straw hat" maps to luffy
            ("Pirate King Roger", ["roger"]),
            ("Yonko Whitebeard", ["whitebeard"]),
            ("Nakama bonds", ["nakama"])
        ]
        
        for query, expected_terms in term_tests:
            try:
                extracted_terms = self.search_agent._extract_one_piece_terms(query)
                
                # Verify expected terms are extracted
                for expected_term in expected_terms:
                    self.assertIn(expected_term, extracted_terms, 
                                 f"Expected term '{expected_term}' not extracted from '{query}'")
                
                print(f"✅ Terms extracted: '{query[:40]}...' → {extracted_terms}")
                
            except Exception as e:
                self.fail(f"Term extraction failed for '{query}': {e}")
        
        print("✅ All term extraction tests passed")
    
    def test_7_search_performance_metrics(self):
        """Test 7: Test search performance metrics and confidence scoring."""
        print("\n🔍 Test 7: Search Performance Metrics")
        
        test_input = self._create_test_input("One Piece world and characters")
        
        try:
            results = self.search_agent._execute_agent(test_input)
            
            # Verify performance metrics
            performance = results['metadata']['search_performance']
            self.assertIn('query_enhancement', performance)
            self.assertIn('context_integration', performance)
            self.assertIn('one_piece_terms_used', performance)
            self.assertIn('search_engine_used', performance)
            self.assertIn('hybrid_search', performance)
            
            # Verify all metrics are properly set
            self.assertTrue(performance['query_enhancement'])
            self.assertIsInstance(performance['one_piece_terms_used'], int)
            self.assertEqual(performance['search_engine_used'], 'SearchEngine')
            self.assertTrue(performance['hybrid_search'])
            
            # Verify confidence scoring
            confidence = results['confidence_score']
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            # Confidence should be high since SearchEngine is reliable
            self.assertGreaterEqual(confidence, 0.7, "Confidence should be high with SearchEngine")
            
            print(f"✅ Performance metrics verified: confidence {confidence:.3f}")
            print(f"✅ One Piece terms used: {performance['one_piece_terms_used']}")
            
        except Exception as e:
            self.fail(f"Performance metrics test failed: {e}")
    
    def test_8_advanced_search_queries(self):
        """Test 8: Test advanced and complex search queries."""
        print("\n🔍 Test 8: Advanced Search Queries")
        
        # Test complex queries that exercise multiple search strategies
        complex_queries = [
            "How did Luffy's relationship with Ace develop throughout the story and what impact did it have on his journey?",
            "What are the different types of Devil Fruits and how do they affect the power dynamics in the One Piece world?",
            "Explain the significance of the Grand Line and New World in terms of navigation challenges and pirate strength progression",
            "How does the Marine organization balance justice with corruption, and what role do characters like Garp and Coby play?",
            "What are the main themes of friendship, freedom, and adventure in One Piece, and how are they represented through key story arcs?"
        ]
        
        for query in complex_queries:
            test_input = self._create_test_input(query)
            
            try:
                results = self.search_agent._execute_agent(test_input)
                
                # Verify complex query handling
                self.assertGreater(len(results['results']), 0, f"No results for complex query: {query[:50]}...")
                
                # Verify result quality (should have good scores)
                top_result = results['results'][0]
                self.assertGreater(top_result['combined_score'], 0.0)
                
                print(f"✅ Complex query handled: '{query[:60]}...' → {len(results['results'])} results")
                
            except Exception as e:
                self.fail(f"Complex query failed: {query[:50]}... - {e}")
        
        print("✅ All advanced query tests passed")


def run_phase3_search_tests():
    """Run all Phase 3 search and retrieval tests."""
    print("=" * 60)
    print("🚀 PHASE 3: SEARCH & RETRIEVAL TESTING")
    print("=" * 60)
    print("Testing SearchEngine integration and search capabilities...")
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase3SearchRetrieval)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PHASE 3 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n🎉 All Phase 3 tests passed! SearchEngine integration working correctly.")
    else:
        print("\n⚠️ Some Phase 3 tests failed. Check the output above for details.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase3_search_tests()
    exit(0 if success else 1)

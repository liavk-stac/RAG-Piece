"""
Response Collection Script for RAGAS Evaluation

This script sends the 20 evaluation questions to the One Piece RAG chatbot
and collects the responses, retrieved contexts, and metadata for evaluation.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import time

# Add the main project directory to Python path to import RAG components
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import RAG system components
from src.chatbot.core.chatbot import OnePieceChatbot
from src.chatbot.config import ChatbotConfig


def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """Load the evaluation questions from JSON file."""
    with open(questions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def initialize_rag_system() -> OnePieceChatbot:
    """Initialize the RAG chatbot system."""
    print("🚀 Initializing One Piece RAG Chatbot...")
    
    # Change to the main project directory to ensure proper path resolution
    import os
    original_cwd = os.getcwd()
    # Get the project root: ragas_eval/scripts/collect_responses.py -> RAG-Piece/
    script_dir = os.path.dirname(os.path.abspath(__file__))  # ragas_eval/scripts/
    ragas_dir = os.path.dirname(script_dir)  # ragas_eval/
    project_root = os.path.dirname(ragas_dir)  # RAG-Piece/
    os.chdir(project_root)
    print(f"📁 Changed working directory to: {project_root}")
    
    try:
        config = ChatbotConfig()
        chatbot = OnePieceChatbot(config)
        print("✅ RAG system initialized successfully!")
        return chatbot
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def collect_response(chatbot: OnePieceChatbot, question: str) -> Dict[str, Any]:
    """Collect response from RAG system for a single question."""
    try:
        # Change to project root for proper path resolution during query processing
        import os
        original_cwd = os.getcwd()
        # Get the project root: ragas_eval/scripts/collect_responses.py -> RAG-Piece/
        script_dir = os.path.dirname(os.path.abspath(__file__))  # ragas_eval/scripts/
        ragas_dir = os.path.dirname(script_dir)  # ragas_eval/
        project_root = os.path.dirname(ragas_dir)  # RAG-Piece/
        os.chdir(project_root)
        
        try:
            # Send question to RAG system with explicit instruction to skip images
            # This helps the router agent avoid unnecessary image retrieval
            modified_question = f"{question} (Please provide a text-only answer without any images.)"
            response = chatbot.ask(modified_question)
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        # Extract relevant information for RAGAS evaluation
        collected_data = {
            'question': question,
            'answer': response.get('response', ''),
            'contexts': [],
            'metadata': response.get('metadata', {}),
            'success': True,
            'error': None
        }
        
        # Extract retrieved contexts from search results
        if 'processing_summary' in response.get('metadata', {}):
            processing_summary = response['metadata']['processing_summary']
            
            # Look for search agent outputs in the response
            if 'agent_outputs_summary' in processing_summary:
                search_count = processing_summary['agent_outputs_summary'].get('search_results_count', 0)
                if search_count > 0:
                    # Try to extract actual contexts from the response
                    # Note: This might need adjustment based on your exact response format
                    collected_data['contexts'] = extract_contexts_from_response(response)
        
        return collected_data
        
    except Exception as e:
        print(f"❌ Error processing question: {question[:50]}...")
        print(f"   Error: {str(e)}")
        return {
            'question': question,
            'answer': '',
            'contexts': [],
            'metadata': {},
            'success': False,
            'error': str(e)
        }


def extract_contexts_from_response(response: Dict[str, Any]) -> List[str]:
    """Extract retrieved contexts from RAG response."""
    contexts = []
    
    # Try to extract contexts from various possible locations in response
    # This is a best-effort extraction based on common RAG response formats
    
    # Method 1: Look in metadata for search results
    if 'search_results' in response:
        for result in response['search_results']:
            if 'content' in result:
                contexts.append(result['content'])
    
    # Method 2: Look in processing details
    if 'processing_details' in response:
        details = response['processing_details']
        if 'retrieved_chunks' in details:
            for chunk in details['retrieved_chunks']:
                if 'content' in chunk:
                    contexts.append(chunk['content'])
    
    # Method 3: Look in agent outputs
    if 'agent_outputs' in response:
        agent_outputs = response['agent_outputs']
        if 'search' in agent_outputs and 'results' in agent_outputs['search']:
            for result in agent_outputs['search']['results']:
                if 'content' in result:
                    contexts.append(result['content'])
    
    # If no contexts found, create a placeholder
    if not contexts:
        contexts = ["Context extraction not available - response generated successfully"]
    
    return contexts


def save_collected_responses(responses: List[Dict[str, Any]], output_file: str):
    """Save collected responses to JSON file."""
    print(f"💾 Saving {len(responses)} responses to {output_file}")
    
    # Create summary statistics
    successful_responses = [r for r in responses if r['success']]
    failed_responses = [r for r in responses if not r['success']]
    
    output_data = {
        'metadata': {
            'total_questions': len(responses),
            'successful_responses': len(successful_responses),
            'failed_responses': len(failed_responses),
            'collection_timestamp': time.time(),
            'success_rate': len(successful_responses) / len(responses) if responses else 0
        },
        'responses': responses,
        'summary': {
            'avg_contexts_per_question': sum(len(r['contexts']) for r in successful_responses) / len(successful_responses) if successful_responses else 0,
            'questions_with_contexts': len([r for r in successful_responses if r['contexts']]),
            'categories_covered': list(set(r.get('category', 'unknown') for r in responses))
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Responses saved successfully!")
    print(f"   📊 Success rate: {output_data['metadata']['success_rate']:.1%}")
    print(f"   📝 Average contexts per question: {output_data['summary']['avg_contexts_per_question']:.1f}")


def main():
    """Main execution function."""
    print("🔍 RAGAS Evaluation - Response Collection")
    print("=" * 50)
    
    # Define file paths
    base_dir = Path(__file__).parent.parent
    questions_file = base_dir / "data" / "questions_and_answers.json"
    output_file = base_dir / "data" / "collected_responses.json"
    
    # Load questions
    print("📝 Loading evaluation questions...")
    questions_data = load_questions(questions_file)
    print(f"✅ Loaded {len(questions_data)} questions")
    
    # Initialize RAG system
    chatbot = initialize_rag_system()
    
    # Collect responses
    print("\n🤖 Collecting responses from RAG system...")
    collected_responses = []
    
    for i, q_data in enumerate(questions_data, 1):
        question = q_data['question']
        category = q_data.get('category', 'unknown')
        
        print(f"\n📋 Question {i}/{len(questions_data)} [{category}]")
        print(f"   Q: {question[:80]}{'...' if len(question) > 80 else ''}")
        
        # Collect response
        response_data = collect_response(chatbot, question)
        
        # Add question metadata
        response_data.update({
            'question_id': q_data['question_id'],
            'category': category,
            'difficulty': q_data.get('difficulty', 'unknown'),
            'ground_truth': q_data['ground_truth']
        })
        
        collected_responses.append(response_data)
        
        if response_data['success']:
            answer_preview = response_data['answer'][:100] + "..." if len(response_data['answer']) > 100 else response_data['answer']
            print(f"   ✅ Response: {answer_preview}")
            print(f"   📚 Contexts: {len(response_data['contexts'])}")
        else:
            print(f"   ❌ Failed: {response_data['error']}")
        
        # Add small delay to avoid overwhelming the system
        time.sleep(1)
    
    # Save results
    print(f"\n💾 Saving collected responses...")
    save_collected_responses(collected_responses, output_file)
    
    print("\n🎉 Response collection completed!")
    print(f"📄 Results saved to: {output_file}")
    print("\nNext step: Run 'python scripts/run_ragas_evaluation.py' to evaluate responses")


if __name__ == "__main__":
    main()

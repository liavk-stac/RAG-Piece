"""
RAGAS Evaluation Script

This script runs RAGAS evaluation on the collected responses using the
four key metrics: Answer Correctness, Context Precision, Context Recall, and Faithfulness.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import time
import pandas as pd
from datasets import Dataset

# Import RAGAS components
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def load_collected_data(data_file: str) -> Dict[str, Any]:
    """Load the collected responses from the collection script."""
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_ragas_dataset(collected_data: Dict[str, Any]) -> Dataset:
    """Prepare data in RAGAS-compatible format."""
    print("🔄 Preparing dataset for RAGAS evaluation...")
    
    responses = collected_data['responses']
    successful_responses = [r for r in responses if r['success']]
    
    print(f"📊 Processing {len(successful_responses)}/{len(responses)} successful responses")
    
    # Prepare data in RAGAS format
    ragas_data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': [],
        'question_id': [],
        'category': [],
        'difficulty': []
    }
    
    for response in successful_responses:
        ragas_data['question'].append(response['question'])
        ragas_data['answer'].append(response['answer'])
        ragas_data['contexts'].append(response['contexts'])
        ragas_data['ground_truth'].append(response['ground_truth'])
        ragas_data['question_id'].append(response['question_id'])
        ragas_data['category'].append(response.get('category', 'unknown'))
        ragas_data['difficulty'].append(response.get('difficulty', 'unknown'))
    
    # Create RAGAS dataset
    dataset = Dataset.from_dict(ragas_data)
    print(f"✅ Dataset prepared with {len(dataset)} samples")
    
    return dataset


def run_ragas_evaluation(dataset: Dataset) -> Dict[str, Any]:
    """Run RAGAS evaluation with the four key metrics."""
    print("🎯 Running RAGAS evaluation...")
    print("   Metrics: Answer Correctness, Context Precision, Context Recall, Faithfulness")
    
    # Define metrics to evaluate
    metrics = [
        answer_correctness,
        context_precision,
        context_recall,
        faithfulness
    ]
    
    try:
        # Run evaluation
        print("⏳ Evaluating... (this may take several minutes)")
        start_time = time.time()
        
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=None,  # Uses default OpenAI model
            embeddings=None  # Uses default embeddings
        )
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        print(f"✅ Evaluation completed in {evaluation_time:.1f} seconds")
        
        return {
            'evaluation_results': result,
            'evaluation_time': evaluation_time,
            'timestamp': time.time(),
            'metrics_evaluated': [metric.name for metric in metrics],
            'dataset_size': len(dataset)
        }
        
    except Exception as e:
        print(f"❌ Error during RAGAS evaluation: {str(e)}")
        raise


def process_evaluation_results(evaluation_data: Dict[str, Any], dataset: Dataset) -> Dict[str, Any]:
    """Process and organize evaluation results."""
    print("📊 Processing evaluation results...")
    
    results = evaluation_data['evaluation_results']
    
    # Extract overall scores
    overall_scores = {}
    print(f"Debug: results type = {type(results)}")
    print(f"Debug: results keys = {list(results.keys()) if hasattr(results, 'keys') else 'No keys method'}")
    
    # Extract scores using to_pandas() method for RAGAS EvaluationResult
    try:
        df = results.to_pandas()
        print(f"Debug: DataFrame columns = {list(df.columns)}")
        print(f"Debug: DataFrame shape = {df.shape}")
        
        for metric_name in evaluation_data['metrics_evaluated']:
            if metric_name in df.columns:
                # Calculate mean score for each metric
                mean_score = df[metric_name].mean()
                overall_scores[metric_name] = float(mean_score)
                print(f"Debug: {metric_name} = {mean_score}")
    except Exception as e:
        print(f"Debug: Error extracting scores - {e}")
        overall_scores = {"error": f"Could not extract scores: {e}"}
    
    # Create per-question results
    per_question_results = []
    for i in range(len(dataset)):
        question_result = {
            'question_id': dataset[i]['question_id'],
            'question': dataset[i]['question'],
            'category': dataset[i]['category'],
            'difficulty': dataset[i]['difficulty'],
            'scores': {}
        }
        
        # Add individual metric scores for this question from DataFrame
        try:
            for metric_name in evaluation_data['metrics_evaluated']:
                if metric_name in df.columns:
                    question_result['scores'][metric_name] = float(df.iloc[i][metric_name])
        except Exception as e:
            print(f"Debug: Error extracting per-question scores for question {i}: {e}")
            for metric_name in evaluation_data['metrics_evaluated']:
                question_result['scores'][metric_name] = None
        
        per_question_results.append(question_result)
    
    # Calculate category-wise performance
    category_breakdown = {}
    categories = set(item['category'] for item in per_question_results)
    
    for category in categories:
        category_questions = [q for q in per_question_results if q['category'] == category]
        category_scores = {}
        
        for metric_name in evaluation_data['metrics_evaluated']:
            scores = [q['scores'].get(metric_name) for q in category_questions if q['scores'].get(metric_name) is not None]
            if scores:
                category_scores[metric_name] = {
                    'mean': sum(scores) / len(scores),
                    'count': len(scores),
                    'min': min(scores),
                    'max': max(scores)
                }
        
        category_breakdown[category] = {
            'question_count': len(category_questions),
            'scores': category_scores
        }
    
    # Identify best and worst performing questions
    def get_average_score(question_scores):
        scores = [v for v in question_scores.values() if v is not None]
        return sum(scores) / len(scores) if scores else 0
    
    questions_with_avg = [(q, get_average_score(q['scores'])) for q in per_question_results]
    questions_with_avg.sort(key=lambda x: x[1], reverse=True)
    
    best_questions = [q[0] for q in questions_with_avg[:3]]
    worst_questions = [q[0] for q in questions_with_avg[-3:]]
    
    return {
        'overall_scores': overall_scores,
        'per_question_results': per_question_results,
        'category_breakdown': category_breakdown,
        'performance_analysis': {
            'best_performing_questions': best_questions,
            'worst_performing_questions': worst_questions,
            'total_questions_evaluated': len(per_question_results)
        },
        'evaluation_metadata': {
            'evaluation_time': evaluation_data['evaluation_time'],
            'timestamp': evaluation_data['timestamp'],
            'metrics_used': evaluation_data['metrics_evaluated']
        }
    }


def save_evaluation_results(processed_results: Dict[str, Any], output_file: str):
    """Save processed evaluation results to JSON file."""
    print(f"💾 Saving evaluation results to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    overall_scores = processed_results['overall_scores']
    print(f"\n📊 EVALUATION SUMMARY")
    print(f"=" * 40)
    for metric, score in overall_scores.items():
        print(f"   {metric}: {score:.3f}")
    
    print(f"\n📈 CATEGORY BREAKDOWN")
    print(f"=" * 40)
    for category, data in processed_results['category_breakdown'].items():
        print(f"   {category} ({data['question_count']} questions):")
        for metric, scores in data['scores'].items():
            print(f"     {metric}: {scores['mean']:.3f}")
    
    print(f"\n✅ Detailed results saved to: {output_file}")


def main():
    """Main execution function."""
    print("🎯 RAGAS Evaluation Runner")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please ensure your .env file contains: OPENAI_API_KEY=your_key_here")
        return
    
    # Define file paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "collected_responses.json"
    output_file = base_dir / "data" / "evaluation_results.json"
    
    # Check if collected responses exist
    if not input_file.exists():
        print(f"❌ Error: Collected responses not found at {input_file}")
        print("   Please run 'python scripts/collect_responses.py' first")
        return
    
    # Load collected data
    print("📂 Loading collected responses...")
    collected_data = load_collected_data(input_file)
    print(f"✅ Loaded data for {collected_data['metadata']['total_questions']} questions")
    print(f"   📊 Success rate: {collected_data['metadata']['success_rate']:.1%}")
    
    # Prepare dataset
    dataset = prepare_ragas_dataset(collected_data)
    
    # Run evaluation
    evaluation_data = run_ragas_evaluation(dataset)
    
    # Process results
    processed_results = process_evaluation_results(evaluation_data, dataset)
    
    # Save results
    save_evaluation_results(processed_results, output_file)
    
    print("\n🎉 RAGAS evaluation completed successfully!")
    print(f"📄 Results saved to: {output_file}")
    print("\nNext step: Run 'python scripts/analyze_results.py' to generate detailed analysis")


if __name__ == "__main__":
    main()

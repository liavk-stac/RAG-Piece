"""
Results Analysis Script for RAGAS Evaluation

This script analyzes the RAGAS evaluation results and generates
comprehensive reports with insights and recommendations.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import time
from datetime import datetime


def load_evaluation_results(results_file: str) -> Dict[str, Any]:
    """Load the RAGAS evaluation results."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_performance_insights(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate insights about system performance."""
    overall_scores = results['overall_scores']
    category_breakdown = results['category_breakdown']
    
    insights = {
        'overall_performance': {},
        'category_insights': {},
        'metric_analysis': {},
        'improvement_areas': [],
        'strengths': []
    }
    
    # Overall performance assessment
    avg_score = sum(overall_scores.values()) / len(overall_scores)
    insights['overall_performance'] = {
        'average_score': avg_score,
        'performance_level': get_performance_level(avg_score),
        'total_questions': results['performance_analysis']['total_questions_evaluated']
    }
    
    # Individual metric analysis
    for metric, score in overall_scores.items():
        performance_level = get_performance_level(score)
        insights['metric_analysis'][metric] = {
            'score': score,
            'performance_level': performance_level,
            'interpretation': get_metric_interpretation(metric, score)
        }
        
        # Identify strengths and weaknesses
        if score >= 0.8:
            insights['strengths'].append(f"Strong {metric.replace('_', ' ')} (score: {score:.3f})")
        elif score < 0.7:
            insights['improvement_areas'].append(f"Improve {metric.replace('_', ' ')} (score: {score:.3f})")
    
    # Category performance insights
    for category, data in category_breakdown.items():
        category_avg = calculate_category_average(data['scores'])
        insights['category_insights'][category] = {
            'average_score': category_avg,
            'question_count': data['question_count'],
            'performance_level': get_performance_level(category_avg),
            'best_metric': get_best_metric(data['scores']),
            'worst_metric': get_worst_metric(data['scores'])
        }
    
    return insights


def get_performance_level(score: float) -> str:
    """Determine performance level based on score."""
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.75:
        return "Good"
    elif score >= 0.65:
        return "Fair"
    elif score >= 0.5:
        return "Needs Improvement"
    else:
        return "Poor"


def get_metric_interpretation(metric: str, score: float) -> str:
    """Provide interpretation for each metric score."""
    interpretations = {
        'answer_correctness': {
            'high': "Answers are factually accurate and complete",
            'medium': "Answers are mostly correct but may lack some details",
            'low': "Answers contain factual errors or are incomplete"
        },
        'context_precision': {
            'high': "Retrieved contexts are highly relevant to questions",
            'medium': "Most retrieved contexts are relevant with some noise",
            'low': "Many retrieved contexts are irrelevant or off-topic"
        },
        'context_recall': {
            'high': "System retrieves all necessary information to answer questions",
            'medium': "System retrieves most relevant information with some gaps",
            'low': "System misses important information needed for complete answers"
        },
        'faithfulness': {
            'high': "Answers are well-grounded in retrieved contexts with minimal hallucination",
            'medium': "Answers are mostly grounded but may contain some unsupported claims",
            'low': "Answers contain significant hallucinations or unsupported information"
        }
    }
    
    if score >= 0.8:
        level = 'high'
    elif score >= 0.6:
        level = 'medium'
    else:
        level = 'low'
    
    return interpretations.get(metric, {}).get(level, "No interpretation available")


def calculate_category_average(scores: Dict[str, Dict[str, float]]) -> float:
    """Calculate average score for a category."""
    all_means = [data['mean'] for data in scores.values() if 'mean' in data]
    return sum(all_means) / len(all_means) if all_means else 0.0


def get_best_metric(scores: Dict[str, Dict[str, float]]) -> str:
    """Get the best performing metric for a category."""
    if not scores:
        return "N/A"
    
    best_metric = max(scores.keys(), key=lambda k: scores[k].get('mean', 0))
    return best_metric.replace('_', ' ').title()


def get_worst_metric(scores: Dict[str, Dict[str, float]]) -> str:
    """Get the worst performing metric for a category."""
    if not scores:
        return "N/A"
    
    worst_metric = min(scores.keys(), key=lambda k: scores[k].get('mean', 0))
    return worst_metric.replace('_', ' ').title()


def generate_recommendations(insights: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    # Overall performance recommendations
    avg_score = insights['overall_performance']['average_score']
    if avg_score < 0.7:
        recommendations.append("🎯 Overall system performance needs significant improvement. Focus on the lowest-scoring metrics first.")
    elif avg_score < 0.8:
        recommendations.append("📈 System shows good performance with room for optimization. Target specific weak areas.")
    else:
        recommendations.append("🎉 Excellent overall performance! Focus on maintaining quality and fine-tuning.")
    
    # Metric-specific recommendations
    metric_scores = {k: v['score'] for k, v in insights['metric_analysis'].items()}
    
    if metric_scores.get('context_precision', 1.0) < 0.7:
        recommendations.append("🔍 Improve context precision by refining search algorithms and relevance scoring.")
    
    if metric_scores.get('context_recall', 1.0) < 0.7:
        recommendations.append("📚 Enhance context recall by increasing search result limits and improving query expansion.")
    
    if metric_scores.get('faithfulness', 1.0) < 0.8:
        recommendations.append("🛡️ Reduce hallucinations by improving prompt engineering and context utilization.")
    
    if metric_scores.get('answer_correctness', 1.0) < 0.75:
        recommendations.append("✅ Improve answer accuracy by enhancing knowledge base quality and response generation.")
    
    # Category-specific recommendations
    category_insights = insights['category_insights']
    worst_category = min(category_insights.keys(), 
                        key=lambda k: category_insights[k]['average_score'])
    
    if category_insights[worst_category]['average_score'] < 0.7:
        recommendations.append(f"📖 Focus on improving {worst_category.replace('_', ' ')} questions - lowest performing category.")
    
    # Add general recommendations
    recommendations.extend([
        "🔄 Consider expanding the evaluation dataset with more diverse questions.",
        "📊 Monitor performance over time to track improvements.",
        "🧪 A/B test different system configurations based on these insights."
    ])
    
    return recommendations


def save_detailed_report(results: Dict[str, Any], insights: Dict[str, Any], 
                        recommendations: List[str], output_file: str):
    """Save detailed JSON report."""
    detailed_report = {
        'evaluation_summary': {
            'timestamp': datetime.fromtimestamp(results['evaluation_metadata']['timestamp']).isoformat(),
            'total_questions': results['performance_analysis']['total_questions_evaluated'],
            'evaluation_time': results['evaluation_metadata']['evaluation_time'],
            'metrics_evaluated': results['evaluation_metadata']['metrics_used']
        },
        'overall_scores': results['overall_scores'],
        'performance_insights': insights,
        'recommendations': recommendations,
        'detailed_results': {
            'per_question_results': results['per_question_results'],
            'category_breakdown': results['category_breakdown'],
            'best_performing_questions': results['performance_analysis']['best_performing_questions'],
            'worst_performing_questions': results['performance_analysis']['worst_performing_questions']
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, indent=2, ensure_ascii=False)


def generate_markdown_summary(results: Dict[str, Any], insights: Dict[str, Any], 
                             recommendations: List[str]) -> str:
    """Generate human-readable markdown summary."""
    
    timestamp = datetime.fromtimestamp(results['evaluation_metadata']['timestamp'])
    
    markdown = f"""# RAGAS Evaluation Summary Report

**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Questions Evaluated:** {results['performance_analysis']['total_questions_evaluated']}  
**Evaluation Time:** {results['evaluation_metadata']['evaluation_time']:.1f} seconds

## 📊 Overall Performance

**Average Score:** {insights['overall_performance']['average_score']:.3f} ({insights['overall_performance']['performance_level']})

| Metric | Score | Performance Level |
|--------|-------|------------------|
"""
    
    for metric, data in insights['metric_analysis'].items():
        metric_name = metric.replace('_', ' ').title()
        markdown += f"| {metric_name} | {data['score']:.3f} | {data['performance_level']} |\n"
    
    markdown += f"""
## 📈 Category Performance

"""
    
    for category, data in insights['category_insights'].items():
        category_name = category.replace('_', ' ').title()
        markdown += f"""### {category_name}
- **Questions:** {data['question_count']}
- **Average Score:** {data['average_score']:.3f} ({data['performance_level']})
- **Best Metric:** {data['best_metric']}
- **Worst Metric:** {data['worst_metric']}

"""
    
    markdown += f"""## 🎯 Key Insights

"""
    
    for metric, data in insights['metric_analysis'].items():
        metric_name = metric.replace('_', ' ').title()
        markdown += f"**{metric_name}:** {data['interpretation']}\n\n"
    
    markdown += f"""## 💡 Recommendations

"""
    
    for i, rec in enumerate(recommendations, 1):
        markdown += f"{i}. {rec}\n"
    
    markdown += f"""
## 🏆 Best Performing Questions

"""
    
    for i, q in enumerate(results['performance_analysis']['best_performing_questions'], 1):
        markdown += f"{i}. **[{q['category'].replace('_', ' ').title()}]** {q['question'][:80]}...\n"
    
    markdown += f"""
## ⚠️ Questions Needing Improvement

"""
    
    for i, q in enumerate(results['performance_analysis']['worst_performing_questions'], 1):
        markdown += f"{i}. **[{q['category'].replace('_', ' ').title()}]** {q['question'][:80]}...\n"
    
    markdown += f"""
---
*Report generated by RAGAS Evaluation System*
"""
    
    return markdown


def main():
    """Main execution function."""
    print("📊 RAGAS Results Analysis")
    print("=" * 50)
    
    # Define file paths
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / "data" / "evaluation_results.json"
    detailed_report_file = base_dir / "results" / "detailed_report.json"
    summary_report_file = base_dir / "results" / "summary_report.md"
    
    # Check if evaluation results exist
    if not results_file.exists():
        print(f"❌ Error: Evaluation results not found at {results_file}")
        print("   Please run 'python scripts/run_ragas_evaluation.py' first")
        return
    
    # Load evaluation results
    print("📂 Loading evaluation results...")
    results = load_evaluation_results(results_file)
    print(f"✅ Loaded results for {results['performance_analysis']['total_questions_evaluated']} questions")
    
    # Generate insights
    print("🧠 Generating performance insights...")
    insights = generate_performance_insights(results)
    
    # Generate recommendations
    print("💡 Generating recommendations...")
    recommendations = generate_recommendations(insights)
    
    # Save detailed report
    print("💾 Saving detailed JSON report...")
    save_detailed_report(results, insights, recommendations, detailed_report_file)
    
    # Generate and save markdown summary
    print("📝 Generating markdown summary...")
    markdown_summary = generate_markdown_summary(results, insights, recommendations)
    
    with open(summary_report_file, 'w', encoding='utf-8') as f:
        f.write(markdown_summary)
    
    # Print summary to console
    print(f"\n🎯 EVALUATION RESULTS SUMMARY")
    print(f"=" * 50)
    print(f"Overall Performance: {insights['overall_performance']['average_score']:.3f} ({insights['overall_performance']['performance_level']})")
    print(f"\nMetric Scores:")
    for metric, data in insights['metric_analysis'].items():
        print(f"  {metric.replace('_', ' ').title()}: {data['score']:.3f} ({data['performance_level']})")
    
    print(f"\n📄 Reports saved:")
    print(f"  📊 Detailed report: {detailed_report_file}")
    print(f"  📝 Summary report: {summary_report_file}")
    
    print(f"\n🎉 Analysis completed successfully!")


if __name__ == "__main__":
    main()

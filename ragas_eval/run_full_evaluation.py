"""
Full RAGAS Evaluation Pipeline

This script runs the complete evaluation pipeline:
1. Collect responses from RAG system
2. Run RAGAS evaluation
3. Analyze results and generate reports
"""

import subprocess
import sys
from pathlib import Path
import time


def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")
        return False


def main():
    """Run the complete evaluation pipeline."""
    print("🎯 RAGAS Evaluation - Full Pipeline")
    print("🔍 This will evaluate your One Piece RAG chatbot using RAGAS metrics")
    print("\nPipeline steps:")
    print("1. 🤖 Collect responses from RAG system (20 questions)")
    print("2. 📊 Run RAGAS evaluation (Answer Correctness, Context Precision, Context Recall, Faithfulness)")
    print("3. 📝 Analyze results and generate reports")
    
    # Confirm execution
    response = input("\n▶️  Continue with evaluation? (y/N): ").lower().strip()
    if response != 'y':
        print("⏹️  Evaluation cancelled.")
        return
    
    # Define script paths
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / "scripts"
    
    scripts = [
        (scripts_dir / "collect_responses.py", "Step 1: Collecting responses from RAG system"),
        (scripts_dir / "run_ragas_evaluation.py", "Step 2: Running RAGAS evaluation"),
        (scripts_dir / "analyze_results.py", "Step 3: Analyzing results and generating reports")
    ]
    
    # Track execution time
    start_time = time.time()
    successful_steps = 0
    
    # Run each script in sequence
    for script_path, description in scripts:
        if not script_path.exists():
            print(f"❌ Error: Script not found: {script_path}")
            break
        
        success = run_script(str(script_path), description)
        if success:
            successful_steps += 1
        else:
            print(f"❌ Pipeline failed at: {description}")
            break
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🏁 EVALUATION PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed steps: {successful_steps}/{len(scripts)}")
    print(f"⏱️  Total execution time: {total_time:.1f} seconds")
    
    if successful_steps == len(scripts):
        print(f"\n🎉 Full evaluation completed successfully!")
        print(f"\n📄 Check these files for results:")
        print(f"  📊 Detailed results: ragas_eval/results/detailed_report.json")
        print(f"  📝 Summary report: ragas_eval/results/summary_report.md")
        print(f"  📈 Raw data: ragas_eval/data/evaluation_results.json")
    else:
        print(f"\n⚠️  Evaluation incomplete. Please check error messages above.")
        print(f"💡 You can run individual scripts manually:")
        print(f"   python scripts/collect_responses.py")
        print(f"   python scripts/run_ragas_evaluation.py")
        print(f"   python scripts/analyze_results.py")


if __name__ == "__main__":
    main()

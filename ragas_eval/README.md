# RAGAS Evaluation for One Piece RAG Chatbot

This folder contains a complete RAGAS evaluation system for testing your One Piece RAG chatbot with 20 carefully crafted questions.

## Quick Start

### 1. Install Dependencies
```bash
cd ragas_eval
pip install -r requirements.txt
```

### 2. Ensure Environment Setup
Make sure your `.env` file contains:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Full Evaluation
```bash
python run_full_evaluation.py
```

This will:
- ✅ Collect responses from your RAG system (20 questions)
- ✅ Run RAGAS evaluation (4 metrics)
- ✅ Generate detailed analysis and reports

## Individual Scripts

If you want to run steps individually:

```bash
# Step 1: Collect responses
python scripts/collect_responses.py

# Step 2: Run RAGAS evaluation  
python scripts/run_ragas_evaluation.py

# Step 3: Analyze results
python scripts/analyze_results.py
```

## Evaluation Metrics

- **Answer Correctness**: How accurate and complete are the answers?
- **Context Precision**: Are retrieved contexts relevant to questions?
- **Context Recall**: Is all necessary information retrieved?
- **Faithfulness**: Are answers grounded in retrieved contexts?

## Question Coverage

20 questions covering:
- **Character Abilities** (6 questions): Luffy, Zoro, Sanji, Don Krieg, Arlong, Alvida
- **Story Events** (5 questions): Key battles and encounters
- **Relationships** (4 questions): Character connections
- **World Building** (3 questions): East Blue, Marines, Grand Line
- **Multi-Character** (2 questions): Complex comparisons

## Output Files

After evaluation, check:
- `results/detailed_report.json` - Complete evaluation data
- `results/summary_report.md` - Human-readable summary
- `data/evaluation_results.json` - Raw RAGAS scores

## Success Criteria

Target scores:
- **Answer Correctness**: > 0.75
- **Context Precision**: > 0.70
- **Context Recall**: > 0.75
- **Faithfulness**: > 0.80

## Troubleshooting

**Error: "OPENAI_API_KEY not found"**
- Ensure your `.env` file contains the API key
- Check that the `.env` file is in the project root

**Error: "Collected responses not found"**
- Run `collect_responses.py` first before evaluation

**Error: "RAG system not accessible"**
- Ensure your chatbot system is properly configured
- Check that all imports work from the project root

## Files Structure

```
ragas_eval/
├── README.md                    # This file
├── RAGAS_EVALUATION_SETUP.mdc   # Detailed setup guide
├── run_full_evaluation.py       # Complete pipeline
├── requirements.txt             # Dependencies
├── data/
│   ├── questions_and_answers.json  # Test questions
│   ├── collected_responses.json    # RAG responses (generated)
│   └── evaluation_results.json     # RAGAS scores (generated)
├── scripts/
│   ├── collect_responses.py     # Get RAG responses
│   ├── run_ragas_evaluation.py  # Run RAGAS evaluation
│   └── analyze_results.py       # Generate reports
└── results/
    ├── detailed_report.json     # Complete results
    └── summary_report.md         # Summary report
```

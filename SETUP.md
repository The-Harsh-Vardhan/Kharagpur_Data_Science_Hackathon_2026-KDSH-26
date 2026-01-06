# Setup and Running Guide

## ✅ Completed Steps

1. ✅ Virtual environment created (`venv/`)
2. ✅ All dependencies installed

## 🔑 Next: Configure Gemini API Key

You need a Google Gemini API key to run the LLM-based contradiction detection.

### Option 1: Set Environment Variable (Recommended)

**PowerShell:**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

### Option 2: Edit the gemini_llm.py file directly

Replace this line in `src/gemini_llm.py`:
```python
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```

With:
```python
genai.configure(api_key="YOUR_API_KEY_HERE")
```

## 🚀 Running the Solution

### Step 1: Train the Model

```powershell
# Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1

# Run training
python run_train.py
```

This will:
- Load both novels from `data/Books/`
- Chunk and embed the novels (takes ~5-10 minutes)
- Process each training example
- Train a logistic regression classifier
- Save the model to `models/classifier.pkl`

**Note:** Training may take significant time due to LLM API calls for each claim-evidence pair.

### Step 2: Run Inference

```powershell
python run_inference.py
```

This will:
- Load the trained classifier
- Process test examples
- Generate predictions
- Save results to `submission.csv`

## 📊 Expected Output Files

- `models/classifier.pkl` - Trained classifier
- `submission.csv` - Final predictions in format:
  ```
  id,label
  101,1
  102,0
  ```

## ⚠️ Important Notes

1. **API Costs**: Each training/test example makes multiple Gemini API calls. Monitor your usage.

2. **Processing Time**: 
   - Novel embedding: ~5-10 minutes (one-time per run)
   - Training: Depends on dataset size and API response time
   - Inference: Similar to training time

3. **Data Structure**: Ensure your CSV files have these columns:
   - `train.csv`: `id`, `backstory`, `book`, `label`
   - `test.csv`: `id`, `backstory`, `book`

4. **Book Names**: The `book` column should contain either:
   - `castaways` (for In Search of the Castaways)
   - `monte_cristo` (for The Count of Monte Cristo)

## 🐛 Troubleshooting

### "No module named 'sentence_transformers'"
Ensure virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

### "GEMINI_API_KEY not found"
Set your API key as shown in the configuration section above.

### Memory Issues
If you encounter memory errors with large novels:
- Reduce `max_tokens` in `chunking.py`
- Reduce `k` (number of retrieved chunks) in the main scripts

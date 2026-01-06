# GitHub Issues for KDSH'26 Project

## 🐛 Bugs / Known Issues

### Issue 1: Memory Issues with Large Novel Processing
**Labels**: bug, performance  
**Priority**: High

**Description**:
When processing very large novels (100k+ words), the embedding step may consume excessive memory, potentially causing crashes on systems with limited RAM.

**Current Behavior**:
- All novel chunks are embedded at once
- Can lead to OOM errors on systems with <8GB RAM

**Expected Behavior**:
- Process embeddings in smaller batches
- Add memory-efficient chunking strategy

**Suggested Fix**:
- Implement streaming/batch processing for embeddings ✅ (partially done)
- Add configuration for batch size based on available memory
- Consider using memory mapping for large text files

---

### Issue 2: No API Rate Limiting or Retry Logic
**Labels**: bug, enhancement  
**Priority**: High

**Description**:
Gemini API calls lack rate limiting and retry logic, which can cause failures during training when API rate limits are hit or temporary errors occur.

**Current Behavior**:
- Direct API calls without retry
- No handling of rate limit errors
- Single failure can crash entire training run

**Expected Behavior**:
- Implement exponential backoff retry logic
- Add rate limiting to respect API quotas
- Graceful handling of temporary API failures

**Suggested Implementation**:
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), 
       stop=stop_after_attempt(3))
def gemini_call(prompt):
    # existing code
```

---

### Issue 3: Missing Data Validation
**Labels**: bug, enhancement  
**Priority**: Medium

**Description**:
No validation of CSV file format or data quality before processing begins.

**Current Behavior**:
- Assumes CSV files have correct columns
- No validation of data types or required fields
- Errors only surface during processing

**Expected Behavior**:
- Validate CSV schema before starting
- Check for missing required columns
- Validate data types and value ranges
- Provide clear error messages for data issues

**Example Validation**:
- Check `book_name` values match available novels
- Verify `content` field is non-empty
- Validate `label` values are in allowed set

---

### Issue 4: No Reproducibility Controls
**Labels**: enhancement  
**Priority**: Medium

**Description**:
Training results are not reproducible due to lack of random seed setting.

**Current Behavior**:
- No random seed configuration
- train/validation split may vary between runs
- Classifier results not fully reproducible

**Suggested Fix**:
```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
```

---

## 🚀 Feature Requests / Enhancements

### Issue 5: Sequential Processing is Slow
**Labels**: enhancement, performance  
**Priority**: High

**Description**:
Claims are processed sequentially, making training very slow. With 80 examples × ~5 claims × 5 evidence chunks, we make ~2000 sequential API calls.

**Current Performance**:
- ~2000+ sequential LLM calls for training
- Estimated time: 20-30 minutes

**Proposed Enhancement**:
- Implement async/parallel API calls
- Batch multiple claims together
- Use threading/asyncio for concurrent requests

**Potential Speedup**: 5-10x faster with 10 concurrent requests

---

### Issue 6: Hardcoded Hyperparameters
**Labels**: enhancement  
**Priority**: Medium

**Description**:
Key hyperparameters are hardcoded in source files, making experimentation difficult.

**Hardcoded Values**:
- Chunk size: 800 tokens
- Overlap: 100 tokens
- Retrieval K: 5
- Embedding batch size: 32
- Contradiction threshold: 0.7

**Proposed Solution**:
Create `config.yaml`:
```yaml
chunking:
  max_tokens: 800
  overlap: 100
  
retrieval:
  top_k: 5
  
embedding:
  model: "all-mpnet-base-v2"
  batch_size: 32
  
classification:
  contradiction_threshold: 0.7
```

---

### Issue 7: No Cost Tracking for API Usage
**Labels**: enhancement  
**Priority**: Medium

**Description**:
No tracking of Gemini API usage or estimated costs during training/inference.

**Proposed Features**:
- Count total API calls made
- Estimate tokens used
- Display cost estimate at end of run
- Warning when approaching rate limits

**Implementation**:
```python
class APITracker:
    def __init__(self):
        self.calls = 0
        self.tokens = 0
    
    def track_call(self, prompt, response):
        self.calls += 1
        self.tokens += len(prompt) + len(response)
    
    def report(self):
        print(f"API Calls: {self.calls}")
        print(f"Estimated Cost: ${self.tokens * 0.00001:.2f}")
```

---

### Issue 8: Missing Unit Tests
**Labels**: testing  
**Priority**: Low

**Description**:
No unit tests for core functionality.

**Proposed Test Coverage**:
- `test_chunking.py`: Test chunk size, overlap, boundary cases
- `test_embeddings.py`: Test embedding dimensions, normalization
- `test_retrieval.py`: Test vector search accuracy
- `test_contradiction.py`: Test LLM response parsing
- `test_aggregation.py`: Test score aggregation logic

---

### Issue 9: Limited Error Messages
**Labels**: enhancement, ux  
**Priority**: Low

**Description**:
Error messages don't provide enough context for debugging.

**Examples**:
- Novel file not found → Should suggest checking data/Books/
- API key error → Should link to API key setup instructions
- CSV format error → Should show expected vs actual columns

---

### Issue 10: No Progress Persistence
**Labels**: enhancement  
**Priority**: Low

**Description**:
If training is interrupted, all progress is lost.

**Proposed Enhancement**:
- Save intermediate results after each training example
- Resume from checkpoint on restart
- Store processed features to avoid re-computation

**Implementation**:
```python
# Save checkpoint after each batch
checkpoint = {
    'features': features,
    'labels': labels,
    'processed_ids': processed_ids
}
pickle.dump(checkpoint, open('checkpoint.pkl', 'wb'))
```

---

## 📚 Documentation Issues

### Issue 11: Missing Function Documentation
**Labels**: documentation  
**Priority**: Low

**Description**:
Many functions lack docstrings explaining parameters, return values, and examples.

**Files Needing Docs**:
- `src/aggregation.py`
- `src/retrieval.py`
- `src/train_classifier.py`

---

### Issue 12: No Example Usage Guide
**Labels**: documentation  
**Priority**: Low

**Description**:
Missing examples for:
- How to add new embedding models
- How to experiment with different classifiers
- How to debug API issues
- How to interpret results

---

## 🔧 Infrastructure

### Issue 13: No CI/CD Pipeline
**Labels**: infrastructure  
**Priority**: Low

**Description**:
No automated testing or deployment pipeline.

**Proposed**:
- GitHub Actions for automated testing
- Linting with `black` and `flake8`
- Type checking with `mypy`

---

### Issue 14: Large Files in Git (Potential)
**Labels**: infrastructure  
**Priority**: Medium

**Description**:
Risk of accidentally committing large files (models, data) despite `.gitignore`.

**Proposed Solution**:
- Add Git LFS support
- Add pre-commit hooks to prevent large file commits
- Document what should never be committed

---

## Copy-Paste Template for GitHub

Each issue above can be copied to GitHub with:
- **Title**: Issue number and title
- **Body**: Description, current/expected behavior, suggested fix
- **Labels**: As specified
- **Assignees**: Assign to team members

---

**Priority Guide**:
- **High**: Affects functionality or user experience significantly
- **Medium**: Important but has workarounds
- **Low**: Nice to have improvements

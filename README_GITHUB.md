# Kharagpur Data Science Hackathon 2026
## Track A — Narrative Consistency Reasoning System

A sophisticated system for determining whether hypothetical character backstories are logically consistent with full-length novels using evidence-grounded reasoning and LLM-based contradiction detection.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd KDSH'26
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your Gemini API key
   # GEMINI_API_KEY=your-actual-api-key-here
   ```

5. **Prepare data**
   - Place novel files in `data/Books/`:
     - `In search of the castaways.txt`
     - `The Count of Monte Cristo.txt`
   - Ensure CSV files are in `data/`:
     - `train.csv`
     - `test.csv`

6. **Train the model**
   ```bash
   python run_train.py
   ```

7. **Run inference**
   ```bash
   python run_inference.py
   ```

---

## 📁 Project Structure

```
KDSH'26/
├── data/
│   ├── Books/
│   │   ├── In search of the castaways.txt
│   │   └── The Count of Monte Cristo.txt
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── aggregation.py          # Evidence score aggregation
│   ├── chunking.py              # Text chunking with overlap
│   ├── contradiction.py         # LLM-based contradiction detection
│   ├── embeddings.py            # Sentence transformer embeddings
│   ├── gemini_llm.py           # Gemini API integration
│   ├── retrieval.py            # FAISS vector store
│   └── train_classifier.py     # Logistic regression training
│
├── models/                      # Saved classifier (generated)
├── venv/                        # Virtual environment (gitignored)
│
├── run_train.py                 # Training pipeline
├── run_inference.py             # Inference pipeline
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
└── SETUP.md                     # Detailed setup guide

```

---

## 🛠️ Tech Stack

- **Python 3.12+**
- **Embedding Model**: `all-mpnet-base-v2` (SentenceTransformers)
- **Vector Search**: FAISS
- **LLM**: Google Gemini Pro
- **Classifier**: Logistic Regression (scikit-learn)
- **Document Processing**: NLTK

---

## 📊 Data Format

### Training CSV (`train.csv`)
Required columns:
- `id`: Unique identifier
- `book_name`: Novel name (e.g., "In Search of the Castaways")
- `content`: Hypothetical backstory text
- `label`: "consistent" or "inconsistent"

### Test CSV (`test.csv`)
Required columns:
- `id`: Unique identifier
- `book_name`: Novel name
- `content`: Hypothetical backstory text

---

## 🔧 How It Works

### System Architecture

1. **Novel Chunking** → Split novels into overlapping 800-token chunks
2. **Embedding & Indexing** → Create semantic embeddings and FAISS index
3. **Backstory Decomposition** → Split backstory into atomic claims (sentences)
4. **Evidence Retrieval** → Retrieve top-5 relevant chunks per claim
5. **Contradiction Detection** → LLM classifies each (claim, evidence) pair
6. **Aggregation** → Combine scores using conservative logic
7. **Classification** → Logistic regression produces final binary prediction

### Key Features

- ✅ Evidence-grounded reasoning (no hallucination)
- ✅ Conservative aggregation (contradictions dominate)
- ✅ Efficient long-context processing
- ✅ Explainable decisions via retrieved evidence

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
GEMINI_API_KEY=your-api-key-here
```

### Hyperparameters

You can modify these in the source files:

- **Chunk size**: 800 tokens (`src/chunking.py`)
- **Chunk overlap**: 100 tokens (`src/chunking.py`)
- **Retrieval K**: 5 chunks (`run_train.py`, `run_inference.py`)
- **Embedding model**: `all-mpnet-base-v2` (`src/embeddings.py`)

---

## 📝 Output

### Training
- Saves classifier to `models/classifier.pkl`
- Prints validation accuracy

### Inference
- Generates `submission.csv` with format:
  ```csv
  id,label
  101,1
  102,0
  ```
  Where `1` = consistent, `0` = inconsistent

---

## 🤝 Team Collaboration

### For New Team Members

1. Clone the repo
2. Follow setup instructions above
3. Get your own Gemini API key
4. Never commit your `.env` file (already in `.gitignore`)

### Git Workflow

```bash
# Pull latest changes
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feature/your-feature-name

# Create pull request on GitHub
```

---

## ⚠️ Important Notes

### API Costs
- Each training/test example makes multiple Gemini API calls
- Monitor your API usage and costs
- Consider using a smaller subset for testing

### Processing Time
- Novel embedding: ~5-10 minutes (one-time per run)
- Training: Depends on dataset size and API latency
- Inference: Similar to training time

### Troubleshooting

**"No module named X"**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**"GEMINI_API_KEY not found"**
- Check that `.env` file exists with your API key
- Ensure `python-dotenv` is installed

**Memory errors**
- Reduce chunk size in `src/chunking.py`
- Reduce retrieval K value in main scripts

---

## 📚 Additional Resources

- [Problem Statement](src/Problem_Statement.md)
- [Detailed Setup Guide](SETUP.md)
- [Gemini API Documentation](https://ai.google.dev/docs)

---

## 📄 License

This project is for academic/competition use.

---

## 🙋 Support

For questions or issues:
1. Check existing GitHub issues
2. Create a new issue with details
3. Contact team members via your communication channels

---

**Built for Kharagpur Data Science Hackathon 2026 - Track A**

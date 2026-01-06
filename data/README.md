# Data Directory

Place your data files here:

## Required Files

### Novels (in `Books/` subdirectory)
- `In search of the castaways.txt`
- `The Count of Monte Cristo.txt`

### CSV Files (in this directory)
- `train.csv` - Training data
- `test.csv` - Test data

## CSV Format

### train.csv
Required columns:
- `id`: Unique identifier
- `book_name`: Novel name
- `content`: Backstory text
- `label`: "consistent" or "inconsistent"

### test.csv
Required columns:
- `id`: Unique identifier
- `book_name`: Novel name
- `content`: Backstory text

---

**Note**: Data files are not committed to Git (see `.gitignore`). Team members should obtain data separately.

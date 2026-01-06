# Contributing to KDSH'26 Project

## Getting Started

1. **Setup your environment**
   - Follow instructions in README_GITHUB.md
   - Get your own Gemini API key
   - Never share or commit API keys

2. **Sync with team**
   ```bash
   git pull origin main
   ```

3. **Create a branch for your work**
   ```bash
   git checkout -b feature/your-name-feature-description
   ```

## Making Changes

### Code Style
- Follow PEP 8 conventions
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and modular

### Testing
Before committing:
- Test your changes locally
- Ensure scripts run without errors
- Verify output format matches requirements

### Commit Guidelines
- Write clear, descriptive commit messages
- Use present tense: "Add feature" not "Added feature"
- Reference issues when applicable

Example:
```bash
git commit -m "Add improved chunking strategy for long texts"
```

## Submitting Changes

1. **Push your branch**
   ```bash
   git push origin feature/your-branch-name
   ```

2. **Create Pull Request**
   - Go to GitHub repository
   - Click "New Pull Request"
   - Describe your changes
   - Request review from teammates

3. **Address feedback**
   - Make requested changes
   - Push updates to same branch
   - PR will update automatically

## File Organization

### Do NOT commit:
- `.env` files (personal API keys)
- `venv/` directory
- `models/*.pkl` files (too large)
- `data/*.csv` files (private data)
- Novel `.txt` files (large files)
- `__pycache__/` directories

### DO commit:
- Source code (`src/*.py`)
- Main scripts (`run_*.py`)
- Documentation (`*.md`)
- Configuration templates (`.env.example`)
- Requirements file

## Need Help?

- Check existing documentation
- Ask in team chat
- Create a GitHub issue for bugs
- Request code review from teammates

---

**Remember**: Never commit sensitive data or API keys!

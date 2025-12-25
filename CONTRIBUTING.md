# Contributing to RAMEN

Thank you for your interest in contributing to RAMEN! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

### 1. Submit Wallpapers
- High-quality wallpapers (minimum 2560×1440)
- Original photography or properly licensed images
- Submit via GitHub Issues with the `wallpaper-submission` label

### 2. Improve the Pipeline
- Bug fixes and performance optimizations
- New data sources or API integrations
- Enhanced filtering or quality scoring algorithms
- Better metadata extraction

### 3. Documentation
- Improve README or other documentation
- Add examples and tutorials
- Fix typos or clarify instructions

### 4. Testing
- Report bugs via GitHub Issues
- Test on different platforms
- Contribute test cases

## 🔧 Development Setup

### Prerequisites
- Python 3.10 or higher
- Git
- API keys for testing (Reddit, Unsplash, Pexels)

### Setup Steps

1. **Fork the repository**
```bash
# Click "Fork" on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/RAMEN.git
cd RAMEN
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install pytest pytest-cov mypy  # Development tools
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your test API credentials
```

5. **Run tests**
```bash
pytest tests/ -v
```

## 📝 Code Standards

### Python Style
- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation
- Add docstrings to all functions and classes
- Use Google-style docstrings
- Include examples for complex functions

### Example
```python
def calculate_quality_score(
    image_path: Path,
    weights: dict[str, float]
) -> float:
    """
    Calculate the overall quality score for a wallpaper.
    
    Args:
        image_path: Path to the image file.
        weights: Dictionary of scoring weights for each component.
    
    Returns:
        Quality score between 0.0 and 1.0.
    
    Example:
        >>> weights = {"visual": 0.4, "composition": 0.3}
        >>> score = calculate_quality_score(Path("image.jpg"), weights)
        >>> print(f"Score: {score:.2f}")
        Score: 0.87
    """
    # Implementation here
    pass
```

### Testing
- Write tests for new features
- Maintain test coverage above 80%
- Use pytest fixtures for common setup
- Mock external API calls

## 🔄 Contribution Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes
- Write clean, documented code
- Follow the code standards above
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run tests
pytest tests/ -v

# Check code coverage
pytest tests/ --cov=. --cov-report=html

# Type checking
mypy *.py

# Lint (optional)
flake8 *.py
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add new quality scoring metric"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference any related issues
- Screenshots for UI changes
- Test results

## 🐛 Reporting Bugs

### Before Submitting
- Check existing issues to avoid duplicates
- Test with the latest version
- Gather relevant information

### Bug Report Template
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With config '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., macOS 13.0]
- Python version: [e.g., 3.10.5]
- RAMEN version: [e.g., 1.0.0]

**Logs**
Attach relevant log files or error messages.
```

## 💡 Feature Requests

We welcome feature suggestions! Please:
1. Check if the feature already exists or is planned
2. Describe the use case and benefits
3. Provide examples if possible
4. Be open to discussion and feedback

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] PR description is complete

## 🎨 Wallpaper Submission Guidelines

### Quality Requirements
- **Resolution**: Minimum 2560×1440 (QHD)
- **Format**: JPG, PNG, or WebP
- **File size**: Under 10MB
- **Content**: No watermarks, excessive text, or NSFW content

### Licensing
- You must own the rights or have permission
- Specify the license (CC0, CC-BY, etc.)
- Provide attribution if required

### Submission Process
1. Create a GitHub Issue with label `wallpaper-submission`
2. Include:
   - Image URL or attachment
   - Title and description
   - Artist/photographer credit
   - License information
   - Source URL (if applicable)

## 🤝 Code Review Process

1. **Automated checks** run on all PRs (tests, linting)
2. **Maintainer review** within 3-5 business days
3. **Feedback and iteration** as needed
4. **Merge** once approved and checks pass

## 📞 Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/yourusername/RAMEN/discussions)
- **Bugs**: Create a [GitHub Issue](https://github.com/yourusername/RAMEN/issues)
- **Chat**: Join our community (link TBD)

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior
- Be respectful and constructive
- Accept feedback gracefully
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information

## 📄 License

By contributing to RAMEN, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to RAMEN! 🎉

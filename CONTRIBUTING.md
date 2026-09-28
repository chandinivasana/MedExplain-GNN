# Contributing to MedExplain-GNN

Thank you for your interest in contributing to **MedExplain-GNN**! We welcome contributions from researchers, machine learning engineers, healthcare informatics specialists, and open-source enthusiasts.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to the project maintainers.

---

## How to Contribute

### 1. Reporting Bugs
- Search existing [Issues](https://github.com/chandinivasana/MedExplain-GNN/issues) to avoid duplicates.
- Submit a detailed bug report using our **Bug Report Template**, including your environment details, steps to reproduce, and stack traces.

### 2. Suggesting Features
- Propose new features via the **Feature Request Template**.
- Explain the clinical or algorithmic rationale (e.g., adding a new GNN architecture like GATv2/GraphSAGE, integrating UMLS embeddings, or improving explainability visualization).

### 3. Submitting Pull Requests
1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/MedExplain-GNN.git
   cd MedExplain-GNN
   ```
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Set up virtual environment & install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r ai_engine/requirements.txt
   pip install -r backend/requirements.txt
   pip install pytest httpx
   ```
4. **Run the test suite before committing**:
   ```bash
   pytest tests/ -v
   ```
5. **Commit your changes using Conventional Commits**:
   - `feat: add GATv2 attention layer support`
   - `fix: handle edge case in symptom tokenizer for compound terms`
   - `docs: add benchmark comparison table to README`
6. **Push and open a Pull Request** against the `main` branch.

---

## Development Guidelines

- **Maintain Test Coverage**: Every new feature or bugfix should include corresponding unit tests in `tests/`.
- **Typing & Linting**: Use Python 3.10+ type hints where appropriate. Follow PEP 8 style conventions.
- **Microservices Integrity**: Ensure changes to the AI engine maintain compatibility with both synchronous (`/process`) and asynchronous (`/api/diagnose/async`) endpoints.
- **Safety First**: Do not remove clinical safety disclaimers or medical research notices.

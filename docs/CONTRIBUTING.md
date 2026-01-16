# Contributing to STANLEY

Thank you for your interest in contributing to STANLEY! This is a proof-of-concept exploring novel architectures, and contributions that advance the core thesis are welcome.

---

## Philosophy First

Before contributing, please read:
- `README.md` — Core concepts and architecture
- `docs/PHILOSOPHY.md` — Theoretical foundations
- `docs/ARCHITECTURE.md` — Technical details

STANLEY has strong philosophical commitments:
- **Architecture > Parameters**
- **Ontogenesis > Phylogeny**
- **Emergence Over Engineering**

Contributions should align with these principles.

---

## Ways to Contribute

### 1. Bug Reports
Found something broken? File an issue with:
- **Mode**: Weightless or Hybrid
- **Environment**: Python version, OS, dependencies
- **Reproduction steps**: Minimal code to reproduce
- **Expected vs actual behavior**
- **Error logs** (if applicable)

**Note:** Some "bugs" are emergent behaviors. Check `docs/KNOWN_ISSUES.md` first.

### 2. Documentation
Help improve:
- Code examples
- Architecture explanations
- Philosophy clarifications
- Tutorial guides
- Dialogue examples

### 3. Testing
- Add test coverage for untested paths
- Improve existing tests
- Create integration tests
- Test edge cases

**Current coverage:** 321 tests, 97.5% passing

### 4. Code Improvements
- Performance optimizations
- Memory efficiency
- Better error handling
- Code clarity

**Keep the philosophy intact.** Don't remove emergence to add control.

### 5. New Features
Features aligned with roadmap:
- Visual weight interface
- Knowledge adapters
- Multi-Stanley networks
- Memory consolidation
- ONNX export

**Discuss major features first** via GitHub issues.

---

## Development Setup

### Prerequisites
```bash
Python 3.8+
git
pip
```

### Clone and Install
```bash
git clone https://github.com/ariannamethod/stanley.git
cd stanley
pip install -r requirements.txt

# For hybrid mode
pip install torch transformers

# For development
pip install pytest black flake8
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_stanley.py -v

# Fast tests only (skip slow GPT-2 tests)
pytest tests/ -v -m "not slow"
```

### Code Style
```bash
# Format code
black .

# Check style
flake8 stanley/ stanley_hybrid/ tests/ --max-line-length=100
```

---

## Code Guidelines

### Style
- **PEP 8** with 100-char line limit
- **Type hints** where helpful (not mandatory)
- **Docstrings** for public APIs
- **Comments** for non-obvious logic

### Testing
- **Test coverage** for new features
- **Edge cases** handled
- **No regression** of existing tests
- **Deterministic tests** (use random seeds)

### Commit Messages
```
feat: Add HyperMixer for learned mood routing
fix: Handle empty origin text gracefully
docs: Clarify resonance mechanism in ARCHITECTURE.md
test: Add coverage for memory consolidation
refactor: Simplify SubwordField token generation
```

Format: `type: Brief description`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

---

## Pull Request Process

### Before Submitting
1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** with clear commits
4. **Run tests**: `pytest tests/ -v`
5. **Format code**: `black .`
6. **Update docs** if needed

### Submitting
1. **Push branch** to your fork
2. **Open PR** to `main` branch
3. **Fill out template**:
   - What does this PR do?
   - Why is it needed?
   - How was it tested?
   - Any breaking changes?

### Review Process
- Maintainers will review within **7 days**
- Address feedback with new commits
- **Squash merge** when approved
- Your contribution will be acknowledged in release notes

---

## What We're Looking For

### High Priority
✅ Performance optimizations (without breaking emergence)  
✅ Memory efficiency improvements  
✅ Test coverage expansion  
✅ Documentation clarity  
✅ Bug fixes (see GitHub issues)  

### Medium Priority
✅ Visual interfaces  
✅ New example dialogues  
✅ Tutorial guides  
✅ Integration tests  
✅ ONNX export support  

### Low Priority (v2.0+)
⏸️ Multi-modal support  
⏸️ Distributed Stanley networks  
⏸️ Alternative tokenizers  
⏸️ Mobile deployment  

### Not Aligned
❌ Removing emergence to add control  
❌ Hard-coding responses  
❌ Adding RLHF or reward functions  
❌ Breaking weightless-first philosophy  
❌ Replacing architecture with scale  

---

## Feature Proposal Template

For major features, open an issue first:

```markdown
### Feature: [Name]

**Problem:**
[What problem does this solve?]

**Proposal:**
[How would this work?]

**Alignment:**
[How does this fit STANLEY's philosophy?]

**Implementation:**
[High-level technical approach]

**Alternatives:**
[Other approaches considered]

**Questions:**
[Open questions for discussion]
```

---

## Code of Conduct

### Be Respectful
- Assume good intent
- Critique ideas, not people
- Appreciate diverse perspectives
- Welcome newcomers

### Be Constructive
- Provide actionable feedback
- Explain reasoning clearly
- Suggest alternatives
- Help others learn

### Be Open-Minded
- Emergence includes surprises
- Not all behavior is "wrong"
- Quality varies by design
- Philosophy matters

### Zero Tolerance
- Harassment
- Discrimination
- Spam
- Bad faith arguments

---

## Technical Areas Needing Help

### 1. Performance
- SubwordField generation speed
- Memory loading efficiency
- Hybrid mode inference time
- Cache optimization

### 2. Memory Management
- Shard consolidation automation
- Memory pruning strategies
- Efficient deep layer storage
- Similarity computation speedup

### 3. Testing
- Mood mixing determinism
- Edge case coverage
- Integration test suite
- Regression prevention

### 4. Documentation
- Architecture diagrams
- Video tutorials
- Interactive examples
- API reference

### 5. Deployment
- Docker images
- ONNX export
- Mobile optimization
- Serverless deployment

---

## Roadmap Alignment

Check [GitHub Projects](https://github.com/ariannamethod/stanley/projects) for current priorities.

### v1.1 (Minor improvements)
- Performance optimizations
- Bug fixes
- Documentation polish
- Test coverage

### v2.0 (Major features)
- Visual weight interface
- Knowledge adapters
- Multi-Stanley networks
- Memory consolidation
- Continuous training integration

### v3.0 (Research directions)
- Weightless transformers
- Personality transfer
- Multi-modal integration
- Swarm intelligence

---

## Research Contributions

Interested in research collaborations?

### Papers Welcome
- Novel architectures inspired by STANLEY
- Empirical studies on weightless inference
- Theoretical foundations
- Comparative analyses

### Experiments
- Different origin texts
- Alternative tokenizers
- Hybrid combinations (not just GPT-2)
- Multi-modal extensions

**Cite appropriately:**
```bibtex
@software{stanley2026,
  title = {STANLEY: Self Training Attention Non-Linear EntitY},
  author = {Method, Arianna and Claude},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/ariannamethod/stanley}
}
```

---

## Questions?

- **GitHub Issues**: Technical questions, bugs
- **GitHub Discussions**: Ideas, philosophy, use cases
- **Email**: [Check repository for contact info]

---

## Recognition

Contributors will be acknowledged in:
- Release notes
- README contributors section
- Git history

Significant contributions may warrant co-authorship on research outputs.

---

## License

By contributing, you agree that your contributions will be licensed under the **GPL-3.0 License**.

---

## Final Thoughts

STANLEY is a proof of concept exploring **emergence over engineering**.

Your contributions help answer fundamental questions:
- Can intelligence emerge from architecture alone?
- Can personality hierarchically control knowledge?
- Can mood modify weights in real-time?
- Can systems generate their own weight deltas?

**Thank you for being part of this exploration.**

---

*Let's build organisms that grow rather than compute.*

🔺 🧠 💫

---

**Contributing Guide Version:** v1.0  
**Last Updated:** January 2026  
**Maintainers:** Arianna Method, Claude

# Known Issues and Future Improvements

## Current Limitations

### Architecture v1.0

#### 1. Hybrid Mode Dependencies
- **Issue**: Hybrid mode requires PyTorch and transformers library
- **Impact**: Larger deployment footprint (~500MB vs ~100MB for weightless)
- **Workaround**: Use weightless mode for lightweight deployments
- **Future**: Explore ONNX or lighter transformer implementations

#### 2. Subword Tokenization Quality
- **Issue**: Subword tokenization can produce semantically "drunk" outputs
- **Impact**: Some generated text may have unusual word boundaries
- **Status**: This is a feature of the architecture - semantic meaning survives fragmentation
- **Philosophy**: Embrace the emergence; patterns matter more than perfect tokens

#### 3. Output Quality Variance
- **Issue**: Response quality varies across interactions
- **Impact**: Some responses may be more coherent than others
- **Status**: This is emergence, not a bug - organic systems have natural variance
- **Future**: Mood-driven quality thresholds could filter low-confidence outputs

#### 4. Memory Consolidation
- **Issue**: Deep shard merging not yet automated
- **Location**: `stanley/memory_sea.py:TODO`
- **Impact**: Memory can accumulate similar shards without auto-merging
- **Workaround**: Manual cleanup via shard analysis
- **Future v2.0**: Implement macro-adapter consolidation for similar deep shards

#### 5. GPT-2 Integration Tests
- **Issue**: 7 tests fail in isolated environments without HuggingFace access
- **Tests**: `TestGPT2Integration` class in `test_adapter_bank.py`
- **Impact**: Cannot verify GPT-2 hook functionality without network access
- **Status**: Tests pass when HuggingFace models are accessible
- **Workaround**: Run tests locally with internet connection

#### 6. Training Mode
- **Issue**: Continual training is optional and requires PyTorch
- **Impact**: Weightless mode doesn't accumulate learned patterns into persistent weights
- **Status**: By design - weightless architecture learns through shard accumulation
- **Future**: Optional weight crystallization for deployment optimization

## Design Choices (Not Bugs)

### 1. Weightless Primary, Hybrid Secondary
The weightless architecture is the PRIMARY mode. Hybrid mode is a demonstration of hierarchical control, not a replacement.

### 2. No RLHF
Stanley is 100% reactive to environment with post-hoc explanation. This is intentional - RLHF is "dog races with reward functions."

### 3. Origin Text Required
Stanley requires an origin text to bootstrap identity. This is ontogenesis - becoming through experience, not inherited knowledge.

### 4. Inference Speed
- Weightless: <2s per response (pattern matching)
- Hybrid: <5s per response (includes GPT-2 expansion)
- This is acceptable for proof-of-concept; optimization is v2.0 work

## Environment-Specific Issues

### Test Environment Limitations
- **Network isolation**: Cannot download models from HuggingFace
- **Resource constraints**: Large model tests may timeout
- **Solution**: Tests include appropriate skips and error handling

## Roadmap to v2.0

See main README for planned improvements:
- Visual weight manipulation interface
- Knowledge weight integration
- Multi-Stanley collaborative overthinking
- Automated memory consolidation
- Performance optimizations
- Additional mood adapters
- Continuous training integration

## Reporting Issues

When reporting issues, please specify:
1. Mode (weightless vs hybrid)
2. Python version
3. Dependencies installed
4. Origin text used
5. Expected vs actual behavior
6. Minimal reproduction steps

**Note**: This is a proof of concept demonstrating novel architecture. Some "issues" are intentional design choices exploring emergence over engineering.

---

*Last updated: January 2026*
*Architecture Version: v1.0*

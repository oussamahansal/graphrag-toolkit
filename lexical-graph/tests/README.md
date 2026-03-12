# Lexical-Graph Testing Guide

## Overview

This directory contains the comprehensive unit test suite for the lexical-graph package, a core component of the AWS GraphRAG Toolkit. The test suite verifies functionality across all major modules including indexing, retrieval, storage, configuration, and utility components.

The testing infrastructure combines traditional unit tests with property-based testing using hypothesis to ensure robust quality assurance. All tests are designed to run without external dependencies (AWS services, network access) using comprehensive mocking strategies.

**Key Features:**
- 🧪 Comprehensive unit test coverage across all modules
- 🔄 Property-based testing for invariant verification
- 🎭 Complete AWS service mocking (Bedrock, Neptune, OpenSearch)
- ⚡ Fast execution without external dependencies
- 📊 Detailed coverage reporting with module-specific targets
- 🔁 Async testing support for asynchronous operations
- 🚀 CI/CD integration with automated quality gates

## Prerequisites

- **Python**: >= 3.10
- **Test Framework**: pytest >= 7.0.0
- **Coverage Tool**: pytest-cov >= 4.0.0
- **Mocking**: pytest-mock >= 3.10.0
- **Async Testing**: pytest-asyncio >= 0.21.0
- **Property Testing**: hypothesis >= 6.0.0

## Installation

Install test dependencies using uv (recommended):

```bash
cd lexical-graph

# Install all test dependencies
uv pip install pytest pytest-cov pytest-mock pytest-asyncio hypothesis

# Or install from optional dependencies
uv pip install -e ".[test]"

# Also install the package dependencies
uv pip install -r src/graphrag_toolkit/lexical_graph/requirements.txt
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with extra verbose output (show test names)
pytest tests/ -vv

# Run with short traceback format
pytest tests/ --tb=short

# Run with line-by-line traceback
pytest tests/ --tb=line
```

### Running Specific Tests

```bash
# Run specific test module
pytest tests/unit/test_config.py

# Run specific test class
pytest tests/unit/test_config.py::TestConfigInitialization

# Run specific test function
pytest tests/unit/test_config.py::TestConfigInitialization::test_initialization_with_defaults

# Run tests matching pattern
pytest tests/ -k "test_config"

# Run tests matching multiple patterns
pytest tests/ -k "test_config or test_metadata"
```

### Coverage Reporting

```bash
# Run tests with coverage report
pytest tests/ --cov=src/graphrag_toolkit/lexical_graph --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src/graphrag_toolkit/lexical_graph --cov-report=html

# View HTML report (opens in browser)
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Run with coverage threshold check
pytest tests/ --cov=src/graphrag_toolkit/lexical_graph --cov-fail-under=50

# Generate multiple report formats
pytest tests/ \
  --cov=src/graphrag_toolkit/lexical_graph \
  --cov-report=term-missing \
  --cov-report=html \
  --cov-report=xml
```

### Property-Based Testing

```bash
# Run only property-based tests
pytest tests/ -m property

# Run only unit tests (exclude property tests)
pytest tests/ -m "not property"

# Run property tests with more examples
pytest tests/ --hypothesis-seed=12345 --hypothesis-show-statistics

# Run with specific hypothesis profile
pytest tests/ --hypothesis-profile=dev
```

### Parallel Execution

```bash
# Install pytest-xdist for parallel execution
uv pip install pytest-xdist

# Run tests in parallel (auto-detect CPU count)
pytest tests/ -n auto

# Run tests with specific number of workers
pytest tests/ -n 4
```

### Debugging Tests

```bash
# Drop into debugger on failure
pytest tests/ --pdb

# Drop into debugger at start of each test
pytest tests/ --trace

# Show local variables in tracebacks
pytest tests/ -l

# Stop after first failure
pytest tests/ -x

# Stop after N failures
pytest tests/ --maxfail=3
```

## Test Structure

The test directory mirrors the source code structure for clear correspondence:

```
tests/
├── conftest.py                          # Shared fixtures and configuration
├── README.md                            # This file
├── data/                                # Test data files (if needed)
└── unit/                                # Unit tests
    ├── __init__.py
    ├── test_config.py                   # Configuration management tests
    ├── test_errors.py                   # Custom exception tests
    ├── test_lexical_graph_index.py      # Indexing operations tests
    ├── test_lexical_graph_query_engine.py  # Query engine tests
    ├── test_metadata.py                 # Metadata handling tests
    ├── test_tenant_id.py                # Tenant ID tests
    ├── test_versioning.py               # Version management tests
    ├── indexing/                        # Indexing module tests
    │   ├── __init__.py
    │   ├── test_id_generator.py         # ID generation tests
    │   ├── test_node_handler.py         # Node operations tests
    │   ├── build/                       # Graph construction tests
    │   ├── extract/                     # Document extraction tests
    │   ├── load/                        # Data loading tests
    │   └── utils/                       # Indexing utilities tests
    ├── retrieval/                       # Retrieval module tests
    │   ├── __init__.py
    │   ├── processors/                  # Query processing tests
    │   ├── post_processors/             # Result post-processing tests
    │   ├── retrievers/                  # Graph traversal tests
    │   ├── query_context/               # Context management tests
    │   ├── summary/                     # Summarization tests
    │   └── utils/                       # Retrieval utilities tests
    ├── storage/                         # Storage module tests
    │   ├── __init__.py
    │   ├── test_graph_store_factory.py  # Graph store factory tests
    │   ├── test_vector_store_factory.py # Vector store factory tests
    │   ├── graph/                       # Graph store implementation tests
    │   └── vector/                      # Vector store implementation tests
    └── utils/                           # Utility module tests
        ├── __init__.py
        ├── test_arg_utils.py            # Argument utilities tests
        ├── test_bedrock_utils.py        # Bedrock utilities tests
        ├── test_io_utils.py             # I/O utilities tests
        └── test_llm_cache.py            # LLM cache tests
```

## Fixture Architecture

Fixtures provide reusable test setup code and are defined in `conftest.py`. They are automatically available to all tests.

### Core Fixtures

#### AWS Service Mocks

**`mock_bedrock_client`** - Mock Bedrock LLM client
```python
def test_llm_generation(mock_bedrock_client):
    """Test using mocked Bedrock client."""
    response = mock_bedrock_client.converse(
        messages=[{"role": "user", "content": "test"}]
    )
    assert response['output']['message']['content'][0]['text'] == 'Mock LLM response'
```

**`mock_neptune_store`** - Mock Neptune graph store
```python
def test_graph_query(mock_neptune_store):
    """Test using mocked Neptune store."""
    schema = mock_neptune_store.get_schema()
    assert 'node_types' in schema
    
    results = mock_neptune_store.query("g.V().limit(10)")
    assert isinstance(results, list)
```

**`mock_opensearch_store`** - Mock OpenSearch vector store
```python
def test_vector_search(mock_opensearch_store):
    """Test using mocked OpenSearch store."""
    results = mock_opensearch_store.similarity_search(
        query_vector=[0.1] * 384,
        k=5
    )
    assert len(results) > 0
    assert 'score' in results[0]
```

#### Test Data Fixtures

**`sample_documents`** - Sample documents for indexing tests
```python
def test_document_processing(sample_documents):
    """Test using sample documents."""
    assert len(sample_documents) > 0
    assert 'id' in sample_documents[0]
    assert 'content' in sample_documents[0]
```

**`sample_graph_structure`** - Sample graph structures
```python
def test_graph_traversal(sample_graph_structure):
    """Test using sample graph structure."""
    nodes = sample_graph_structure['nodes']
    edges = sample_graph_structure['edges']
    assert len(nodes) > 0
    assert len(edges) > 0
```

**`sample_tenant_ids`** - Sample tenant IDs
```python
def test_multi_tenancy(sample_tenant_ids):
    """Test using sample tenant IDs."""
    for tenant_id in sample_tenant_ids:
        generator = IDGenerator(tenant_id)
        assert generator.generate().startswith(tenant_id)
```

#### Property Testing Fixtures

**`hypothesis_strategies`** - Hypothesis strategies for property tests
```python
def test_with_strategies(hypothesis_strategies):
    """Test using hypothesis strategies."""
    from hypothesis import given
    
    @given(tenant_id=hypothesis_strategies['tenant_id'])
    def property_test(tenant_id):
        assert len(tenant_id) > 0
    
    property_test()
```

### Fixture Scopes

Fixtures have different scopes that control their lifecycle:

- **function** (default): Created for each test function
- **class**: Created once per test class
- **module**: Created once per test module
- **session**: Created once per test session

```python
@pytest.fixture(scope="session")
def expensive_setup():
    """Expensive setup run once per session."""
    return setup_expensive_resource()

@pytest.fixture(scope="function")
def clean_state():
    """Fresh state for each test."""
    return create_clean_state()
```

## Mocking AWS Services

All AWS service interactions are mocked to ensure tests run without credentials, network access, or costs.

### Bedrock LLM Mocking

```python
def test_llm_call_with_mock(mock_bedrock_client):
    """Test LLM generation with mocked Bedrock."""
    # Mock is automatically configured with deterministic responses
    response = mock_bedrock_client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {"role": "user", "content": [{"text": "Extract entities"}]}
        ]
    )
    
    # Verify mock response structure
    assert 'output' in response
    assert 'message' in response['output']
    assert 'content' in response['output']['message']
```

### Neptune Graph Store Mocking

```python
def test_graph_operations_with_mock(mock_neptune_store):
    """Test graph operations with mocked Neptune."""
    # Mock provides sample schema
    schema = mock_neptune_store.get_schema()
    assert 'Document' in schema['node_types']
    assert 'HAS_CHUNK' in schema['edge_types']
    
    # Mock provides sample query results
    results = mock_neptune_store.query("g.V().hasLabel('Document')")
    assert len(results) > 0
    assert results[0]['type'] == 'Document'
```

### OpenSearch Vector Store Mocking

```python
def test_vector_operations_with_mock(mock_opensearch_store):
    """Test vector operations with mocked OpenSearch."""
    # Mock add_embeddings operation
    result = mock_opensearch_store.add_embeddings(
        embeddings=[[0.1] * 384],
        ids=["doc1"]
    )
    assert result['status'] == 'success'
    
    # Mock similarity search
    results = mock_opensearch_store.similarity_search(
        query_vector=[0.1] * 384,
        k=5
    )
    assert len(results) > 0
    assert all('score' in r for r in results)
```

### Blocking Real AWS Calls

The `block_aws_calls` fixture (autouse) prevents accidental real AWS calls:

```python
# This is automatically applied to all tests
@pytest.fixture(autouse=True)
def block_aws_calls(monkeypatch):
    """Block all real AWS API calls during tests."""
    def mock_boto3_client(*args, **kwargs):
        raise RuntimeError(
            f"Tests must not make real AWS API calls. "
            f"Use mocked clients from conftest.py fixtures. "
            f"Attempted to create client for: {args[0] if args else 'unknown'}"
        )
    
    monkeypatch.setattr('boto3.client', mock_boto3_client)
    monkeypatch.setattr('boto3.resource', mock_boto3_client)
```

## Writing New Tests

### Test Naming Conventions

Follow these naming patterns for consistency:

```python
# Test file names
test_<module_name>.py

# Test class names (optional, for grouping)
class Test<FeatureName>:
    pass

# Test function names
def test_<function_name>_<scenario>():
    pass

# Examples
def test_config_loads_defaults():
    """Verify config loads with default values."""
    pass

def test_config_invalid_chunk_size_raises_error():
    """Verify ValueError raised for invalid chunk size."""
    pass

def test_id_generator_creates_unique_ids():
    """Verify ID generator produces unique IDs."""
    pass
```

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_metadata_serialization():
    """Verify metadata serializes to JSON correctly."""
    # Arrange: Set up test data
    metadata = Metadata({'key': 'value', 'number': 42})
    
    # Act: Perform the operation
    json_str = metadata.to_json()
    
    # Assert: Verify the result
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed == {'key': 'value', 'number': 42}
```

### Testing Error Conditions

Use `pytest.raises` to test exceptions:

```python
def test_validate_tenant_id_empty_raises_error():
    """Verify ValueError raised for empty tenant ID."""
    with pytest.raises(ValueError, match="Tenant ID cannot be empty"):
        validate_tenant_id("")

def test_config_invalid_chunk_size_raises_error():
    """Verify ValueError raised for negative chunk size."""
    config = Config(chunk_size=-1)
    
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        config.validate()
```

### Testing Async Code

Use `@pytest.mark.asyncio` for async tests:

```python
import pytest

@pytest.mark.asyncio
async def test_async_retrieval(mock_graph_store):
    """Test async retrieval operation."""
    retriever = AsyncRetriever(graph_store=mock_graph_store)
    
    result = await retriever.retrieve_async("test query")
    
    assert isinstance(result, list)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_async_with_timeout(mock_graph_store):
    """Test async operation with timeout."""
    retriever = AsyncRetriever(graph_store=mock_graph_store, timeout=1.0)
    
    with pytest.raises(asyncio.TimeoutError):
        await retriever.retrieve_async("slow query")
```

### Parametrized Tests

Use `@pytest.mark.parametrize` for testing multiple scenarios:

```python
@pytest.mark.parametrize("tenant_id,expected", [
    ("tenant-001", "tenant-001"),
    ("TENANT-002", "tenant-002"),
    ("  tenant-003  ", "tenant-003"),
])
def test_format_tenant_id_variations(tenant_id, expected):
    """Test tenant ID formatting with various inputs."""
    result = format_tenant_id(tenant_id)
    assert result == expected

@pytest.mark.parametrize("chunk_size", [-1, 0, 10001])
def test_config_invalid_chunk_sizes(chunk_size):
    """Test config validation rejects invalid chunk sizes."""
    config = Config(chunk_size=chunk_size)
    
    with pytest.raises(ValueError):
        config.validate()
```

## Property-Based Testing

Property-based testing uses hypothesis to verify invariants across many generated inputs.

### Writing Property Tests

```python
from hypothesis import given, strategies as st, settings

@given(tenant_id=st.text(min_size=1, max_size=50))
@settings(max_examples=100)
def test_id_generator_uniqueness_property(tenant_id):
    """
    Property: Generated IDs are unique.
    
    For any tenant ID, generating multiple IDs should produce
    unique values to prevent collisions.
    
    Validates: Requirement 3.1 (ID uniqueness)
    """
    generator = IDGenerator(tenant_id)
    ids = [generator.generate() for _ in range(100)]
    
    assert len(ids) == len(set(ids)), f"Duplicate IDs for tenant {tenant_id}"
```

### Common Hypothesis Strategies

```python
from hypothesis import strategies as st

# Text with constraints
tenant_id_strategy = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='-_'
    )
)

# Integers with range
chunk_size_strategy = st.integers(min_value=100, max_value=10000)

# Lists with constraints
embedding_strategy = st.lists(
    st.floats(min_value=-1.0, max_value=1.0),
    min_size=384,
    max_size=384
)

# Dictionaries
metadata_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=50),
    values=st.one_of(st.text(), st.integers(), st.floats())
)

# Fixed dictionaries (all keys required)
document_strategy = st.fixed_dictionaries({
    'id': st.uuids().map(str),
    'title': st.text(min_size=1, max_size=200),
    'content': st.text(min_size=10, max_size=10000)
})
```

### Property Test Examples

**Idempotence Property:**
```python
@given(tenant_id=tenant_id_strategy)
@settings(max_examples=100)
def test_format_tenant_id_idempotence_property(tenant_id):
    """Property: Formatting is idempotent."""
    formatted_once = format_tenant_id(tenant_id)
    formatted_twice = format_tenant_id(formatted_once)
    assert formatted_once == formatted_twice
```

**Round-trip Property:**
```python
@given(data=metadata_strategy)
@settings(max_examples=100)
def test_metadata_json_roundtrip_property(data):
    """Property: JSON serialization is reversible."""
    metadata = Metadata(data)
    serialized = metadata.to_json()
    deserialized = Metadata.from_json(serialized)
    assert deserialized.to_dict() == metadata.to_dict()
```

**Invariant Property:**
```python
@given(results=st.lists(st.dictionaries(keys=st.text(), values=st.text())))
@settings(max_examples=100)
def test_filtering_reduces_count_property(results):
    """Property: Filtering reduces or maintains result count."""
    filtered = filter_results(results, lambda x: True)
    assert len(filtered) <= len(results)
```

## Coverage Targets

Coverage targets vary by module complexity:

| Module Type | Target Coverage | Rationale |
|-------------|-----------------|-----------|
| Utility modules (utils/, tenant_id.py, id_generator.py) | 80% | Simple, deterministic functions |
| Core modules (config.py, errors.py, metadata.py, versioning.py) | 75% | Configuration and data structures |
| Indexing modules | 70% | Mix of algorithms and I/O |
| Retrieval modules | 70% | Complex logic with external dependencies |
| Storage modules | 60% | Heavy AWS service interaction |
| Query engines (lexical_graph_index.py, lexical_graph_query_engine.py) | 65% | Orchestration with multiple dependencies |

**Overall Target**: 50% (enforced in CI/CD)

### Checking Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src/graphrag_toolkit/lexical_graph --cov-report=term-missing

# Example output:
# Name                                    Stmts   Miss  Cover   Missing
# ---------------------------------------------------------------------
# src/graphrag_toolkit/lexical_graph/config.py      45      5    89%   23-27
# src/graphrag_toolkit/lexical_graph/metadata.py    32      2    94%   45, 67
# ...
# TOTAL                                  1234    123    90%
```

### Improving Coverage

1. **Identify uncovered lines**: Use `--cov-report=term-missing` to see line numbers
2. **Add tests for missing lines**: Focus on error paths and edge cases
3. **Review exclusions**: Ensure `pragma: no cover` is justified
4. **Check HTML report**: Use `--cov-report=html` for visual analysis

```bash
# Generate HTML report for detailed analysis
pytest tests/ --cov=src/graphrag_toolkit/lexical_graph --cov-report=html
open htmlcov/index.html
```

## Debugging Test Failures

### Common Debugging Commands

```bash
# Show full traceback
pytest tests/unit/test_config.py -v --tb=long

# Show local variables in traceback
pytest tests/unit/test_config.py -l

# Drop into debugger on failure
pytest tests/unit/test_config.py --pdb

# Stop after first failure
pytest tests/unit/test_config.py -x

# Run only failed tests from last run
pytest tests/ --lf

# Run failed tests first, then others
pytest tests/ --ff
```

### Debugging Property Tests

When a property test fails, hypothesis provides a counterexample:

```bash
# Example failure output:
# Falsifying example: test_id_generator_uniqueness_property(
#     tenant_id='a-b'
# )

# Reproduce with specific seed
pytest tests/ --hypothesis-seed=12345

# Show hypothesis statistics
pytest tests/ --hypothesis-show-statistics

# Increase verbosity for hypothesis
pytest tests/ --hypothesis-verbosity=verbose
```

### Debugging Async Tests

```bash
# Enable asyncio debug mode
PYTHONASYNCIODEBUG=1 pytest tests/unit/retrieval/

# Show asyncio warnings
pytest tests/ -W default::DeprecationWarning

# Increase timeout for slow async operations
pytest tests/ --timeout=30
```

### Using Print Debugging

```python
def test_with_debug_output(capfd):
    """Test with captured output."""
    print("Debug: Starting test")
    result = some_function()
    print(f"Debug: Result = {result}")
    
    # Capture output
    captured = capfd.readouterr()
    print(f"Captured stdout: {captured.out}")
    
    assert result == expected
```

## Continuous Integration

Tests run automatically in GitHub Actions on:
- Push to `main` branch (when lexical-graph files change)
- Pull requests to `main` branch (when lexical-graph files change)

### CI Workflow Configuration

The workflow is defined in `.github/workflows/lexical-graph-tests.yml`:

```yaml
name: Lexical Graph Unit Tests

on:
  push:
    branches: [main]
    paths:
      - "lexical-graph/**"
      - ".github/workflows/lexical-graph-tests.yml"
  pull_request:
    branches: [main]
    paths:
      - "lexical-graph/**"
      - ".github/workflows/lexical-graph-tests.yml"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
```

### CI Test Execution

The CI workflow:
1. Tests against Python 3.10, 3.11, and 3.12
2. Installs all dependencies using uv
3. Runs all tests with coverage reporting
4. Fails if any test fails or coverage drops below 50%
5. Generates coverage summary in GitHub step summary
6. Uploads HTML coverage report as artifact (Python 3.12 only)

### Viewing CI Results

- **Test Results**: Check the "Actions" tab in GitHub
- **Coverage Summary**: View in the workflow run summary
- **Coverage Report**: Download artifact from workflow run
- **Failure Details**: Click on failed step for full output

## Test Maintenance

### When to Update Tests

1. **Code Changes**: Update tests when implementation changes
2. **Bug Fixes**: Add regression test for every bug fix
3. **New Features**: Add tests for all new functionality
4. **Refactoring**: Update tests to match new structure
5. **API Changes**: Update tests when interfaces change

### Handling Flaky Tests

Flaky tests (tests that sometimes pass, sometimes fail) should be investigated immediately:

1. **Identify the flake**: Run test multiple times
   ```bash
   pytest tests/unit/test_flaky.py --count=100
   ```

2. **Common causes**:
   - Race conditions in async code
   - Timing dependencies
   - Shared state between tests
   - Non-deterministic behavior
   - External dependencies

3. **Fix strategies**:
   - Add proper async synchronization
   - Use fixtures to ensure clean state
   - Mock non-deterministic behavior
   - Increase timeouts if needed

4. **Temporary skip** (last resort):
   ```python
   @pytest.mark.skip(reason="Flaky test - investigating race condition")
   def test_flaky_behavior():
       pass
   ```

### Adding Tests for New Modules

When adding a new module to lexical-graph:

1. **Create test file**: `tests/unit/path/to/test_new_module.py`
2. **Mirror structure**: Match source code directory structure
3. **Add `__init__.py`**: Ensure test directories have `__init__.py`
4. **Write tests**: Cover main functionality, edge cases, errors
5. **Add fixtures**: Create reusable fixtures in `conftest.py` if needed
6. **Check coverage**: Ensure new module meets coverage target
7. **Update docs**: Add examples to this README if needed

### Regression Tests

When fixing a bug:

1. **Write failing test**: Create test that reproduces the bug
2. **Verify failure**: Ensure test fails before fix
3. **Fix the bug**: Implement the fix
4. **Verify success**: Ensure test passes after fix
5. **Document**: Add comment linking test to bug report/issue

```python
def test_bug_123_tenant_id_validation():
    """
    Regression test for bug #123.
    
    Bug: validate_tenant_id() incorrectly accepted empty strings.
    Fix: Added explicit check for empty strings.
    """
    with pytest.raises(ValueError, match="Tenant ID cannot be empty"):
        validate_tenant_id("")
```

## Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'graphrag_toolkit'`

**Solution**: Set PYTHONPATH to include src directory:
```bash
PYTHONPATH=src pytest tests/
```

Or install package in development mode:
```bash
uv pip install -e .
```

### AWS Credential Errors

**Problem**: `NoCredentialsError: Unable to locate credentials`

**Solution**: Tests should never require AWS credentials. If you see this error:
1. Check that you're using mocked fixtures (`mock_bedrock_client`, etc.)
2. Verify `block_aws_calls` fixture is working
3. Ensure you're not accidentally importing real AWS clients

### Fixture Not Found

**Problem**: `fixture 'mock_bedrock_client' not found`

**Solution**:
1. Ensure `conftest.py` is in the correct location
2. Check that fixture is defined in `conftest.py`
3. Verify fixture name matches exactly (case-sensitive)
4. Ensure `conftest.py` doesn't have syntax errors

### Hypothesis Test Failures

**Problem**: Property test fails with counterexample

**Solution**:
1. **Analyze counterexample**: Understand why it fails
2. **Reproduce**: Use `--hypothesis-seed` to reproduce
3. **Fix code or test**:
   - If code is wrong: Fix the implementation
   - If test is wrong: Adjust the property or strategy
4. **Add unit test**: Add specific unit test for the counterexample

Example:
```python
# Hypothesis found: test_format_tenant_id_property(tenant_id='')
# This revealed that empty strings weren't handled

# Add specific unit test
def test_format_tenant_id_empty_string():
    """Verify empty string handling."""
    with pytest.raises(ValueError):
        format_tenant_id('')
```

### Async Test Timeouts

**Problem**: `asyncio.TimeoutError` in async tests

**Solution**:
1. **Increase timeout**: Add timeout parameter to test
   ```python
   @pytest.mark.asyncio(timeout=30)
   async def test_slow_operation():
       pass
   ```

2. **Check for deadlocks**: Ensure proper async/await usage
3. **Mock slow operations**: Use mocks for external calls
4. **Debug**: Enable asyncio debug mode
   ```bash
   PYTHONASYNCIODEBUG=1 pytest tests/
   ```

### Coverage Not Updating

**Problem**: Coverage report doesn't reflect new tests

**Solution**:
1. **Clear cache**: Remove `.pytest_cache` and `__pycache__`
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   find . -type d -name .pytest_cache -exec rm -rf {} +
   ```

2. **Regenerate report**: Run with `--cov-report=html:htmlcov`
3. **Check paths**: Ensure `--cov=src/graphrag_toolkit/lexical_graph` is correct
4. **Verify imports**: Ensure tests import from correct package

## Resources

### Documentation

- **pytest**: https://docs.pytest.org/
- **pytest-cov**: https://pytest-cov.readthedocs.io/
- **unittest.mock**: https://docs.python.org/3/library/unittest.mock.html
- **hypothesis**: https://hypothesis.readthedocs.io/
- **pytest-asyncio**: https://pytest-asyncio.readthedocs.io/

### GraphRAG Toolkit

- **Main Repository**: https://github.com/awslabs/graphrag-toolkit
- **Documentation**: https://awslabs.github.io/graphrag-toolkit/
- **Lexical Graph Module**: https://awslabs.github.io/graphrag-toolkit/lexical-graph/

### Best Practices

- **Testing Best Practices**: https://docs.pytest.org/en/stable/goodpractices.html
- **Property-Based Testing**: https://hypothesis.works/articles/what-is-property-based-testing/
- **Async Testing**: https://pytest-asyncio.readthedocs.io/en/latest/concepts.html
- **Test Fixtures**: https://docs.pytest.org/en/stable/fixture.html

### Getting Help

- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Contributing**: See CONTRIBUTING.md for contribution guidelines

---

**Last Updated**: 2024
**Maintained By**: AWS GraphRAG Toolkit Team

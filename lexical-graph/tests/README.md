# lexical-graph Tests

## Structure & Organisation

The test tree has two roots that serve different purposes:

```
tests/
├── unit/                          # Primary test suite — mirrors src/ layout
│   ├── conftest.py                # TenantId and IdGenerator fixtures
│   ├── test_config.py
│   ├── test_errors.py
│   ├── test_metadata.py
│   ├── test_versioning.py
│   ├── test_lexical_graph_query_engine.py
│   ├── test_logging.py
│   ├── indexing/
│   │   ├── build/                 # Graph construction pipeline
│   │   ├── extract/               # Chunking, LLM extraction, pipeline wiring
│   │   ├── load/                  # Document readers (file, S3, JSON)
│   │   └── utils/                 # Hash, fact, metadata, pipeline utilities
│   ├── prompts/
│   │   └── conftest.py            # Bedrock and S3 client fixtures scoped to prompts
│   ├── retrieval/
│   │   ├── post_processors/
│   │   ├── processors/
│   │   ├── query_context/
│   │   └── utils/
│   ├── storage/
│   │   ├── graph/                 # Neptune Analytics, Neptune DB, in-memory stores
│   │   └── vector/                # DummyVectorIndex, OpenSearch
│   └── utils/                     # arg_utils, bedrock_utils, io_utils, llm_cache, etc.
└── graphrag_toolkit/
    └── lexical_graph/             # Focused tests for specific cross-cutting concerns
        ├── test_config_resilient_client.py
        ├── test_metadata_datetime.py
        ├── test_pragma_simple.py
        ├── test_tenant_id.py
        ├── indexing/
        ├── prompts/
        ├── storage/
        └── utils/
```

`tests/unit/` is the main suite. Its directory layout mirrors `src/graphrag_toolkit/lexical_graph/` so the correspondence between source and test is unambiguous.

`tests/graphrag_toolkit/` exists for tests that target narrow, specific behaviours that don't fit cleanly into the module-level groupings — for example, `test_config_resilient_client.py` tests `ResilientClient` and `_GraphRAGConfig` internals in isolation, and `test_metadata_datetime.py` covers a specific datetime edge case in metadata handling. Think of it as a place for surgical, focused tests rather than broad module coverage.

### Naming conventions

Test files follow `test_<module_name>.py`. Test classes group by the public surface being tested (`TestVectorStoreFactoryRegister`, `TestVectorStoreFactoryForVectorStore`) rather than by implementation detail. Test functions use `test_<method>_<scenario>` — the scenario part is what matters: it should describe the condition or outcome, not restate the method name.

```python
# Good — scenario is clear
def test_factory_creates_opensearch_store(self): ...
def test_try_create_with_none_returns_none(self): ...

# Avoid — just restates the method
def test_try_create(self): ...
```

Every test function must have a docstring. One sentence is enough; it should state intent, not implementation.

### Fixtures and conftest files

`tests/unit/conftest.py` provides `TenantId` and `IdGenerator` fixtures used across the indexing tests. These construct real objects rather than mocks — `TenantId` and `IdGenerator` are pure value types with no external dependencies, so mocking them would add noise without benefit.

`tests/unit/prompts/conftest.py` provides `mock_bedrock_client` and `mock_s3_client` fixtures scoped to the prompts subtree, plus `patch_bedrock_config` and `patch_s3_config` which patch `AWSConfig._get_or_create_client` at the right level. These live in a local `conftest.py` rather than the root one because the patch target is specific to the prompts module.

There is no top-level `tests/conftest.py`. Fixtures are defined close to where they're used. If you need a fixture in more than two subtrees, add it to `tests/unit/conftest.py`.

---

## What Is Being Tested & Why

### Config (`test_config.py`, `test_config_resilient_client.py`)

`_GraphRAGConfig` manages boto3 sessions, AWS client caching, and region resolution. The tests in `test_config_resilient_client.py` focus on `ResilientClient`, which wraps boto3 clients to handle `ExpiredToken` errors by refreshing the underlying client and retrying. This retry logic is non-trivial and the tests explicitly verify that:

- A single `ExpiredToken` triggers a refresh and retry, not a raise.
- Non-expiry `ClientError` codes (e.g. `AccessDenied`) propagate immediately without retry.
- `SSOTokenLoadError` during client creation is converted to a `RuntimeError` with a human-readable message.

The `aws_region` property is a known source of `NoRegionError` in test environments — it falls back to `boto3.Session().region_name` when no region is configured. Any test that exercises code paths touching `GraphRAGConfig.aws_region` must patch it:

```python
@patch('graphrag_toolkit.lexical_graph.config.GraphRAGConfig.aws_region',
       new_callable=lambda: property(lambda self: 'us-east-1'))
```

Patching `boto3.client` alone is not sufficient.

### Storage — Graph (`tests/unit/storage/graph/`)

Neptune tests mock at the `boto3.Session` and `GraphRAGConfig` level rather than at `boto3.client`, because the Neptune store constructs its client through the config object. The tests cover:

- Both `NeptuneAnalyticsGraphStoreFactory` and `NeptuneDatabaseGraphStoreFactory` connection string formats (`neptune-graph://`, `neptune-db://`, `https://*.neptune.amazonaws.com`).
- `execute_query` result parsing, parameter passing, and empty result handling.
- Error wrapping: all `ClientError` variants are expected to surface as `GraphQueryError`. Tests for `ThrottlingException` verify the retry path exhausts and still raises `GraphQueryError` rather than leaking the raw botocore exception.
- A documented known bug: both factories raise `AttributeError` when passed `None` instead of returning `None`. The tests assert this current behaviour explicitly so any fix is visible as a deliberate test change.

### Storage — Vector (`tests/unit/storage/vector/`, `test_vector_store_factory.py`)

`VectorStoreFactory` tests cover its factory dispatch logic: given a connection string, the right `VectorIndexFactoryMethod` is selected. The `for_composite` tests cover the last-write-wins behaviour when two stores share an index name — this is intentional and the test documents it.

`test_factory_creates_opensearch_store` uses `pytest.importorskip("llama_index.vector_stores.opensearch")` to skip gracefully when the optional OpenSearch dependency isn't installed. It patches both `boto3.client` and `GraphRAGConfig.aws_region` (see above).

### Indexing — Utils (`tests/unit/indexing/utils/`)

`test_hash_utils.py`, `test_hash_utils_property.py`, and `test_hash_utils_performance.py` are intentionally split:

- `test_hash_utils.py` — example-based tests for known inputs and outputs.
- `test_hash_utils_property.py` — Hypothesis-driven property tests. These verify determinism, hex output format, consistent 32-character length (MD5), and collision resistance across arbitrary unicode input. The property tests are the primary guard against silent algorithm changes.
- `test_hash_utils_performance.py` — timing assertions against realistic workloads (10k short strings < 100ms). These are not benchmarks; they're regression guards. The thresholds are intentionally generous to avoid flakiness on CI runners.

### Indexing — Extract (`tests/unit/indexing/extract/`)

The extraction pipeline tests mock the LLM layer entirely. `test_llm_proposition_extractor.py` and `test_batch_llm_proposition_extractor_sync.py` verify that the extractor correctly handles LLM response parsing, including malformed JSON responses and empty outputs — the most likely failure modes in production and the most likely regressions when prompt templates change.

### Prompts (`tests/unit/prompts/`)

The prompt provider tests cover the full provider hierarchy: `StaticPromptProvider`, `FilePromptProvider`, `S3PromptProvider`, `BedrockPromptProvider`, and the factory/registry. `BedrockPromptProvider` tests patch at `AWSConfig._get_or_create_client` rather than at `boto3.client`, because the config layer caches clients and a lower-level patch would be bypassed by the cache.

### Retrieval (`tests/unit/retrieval/`)

Retrieval tests focus on the processor and post-processor pipeline. `test_dedup_results.py` and `test_filter_by_metadata.py` cover the stateful parts of the pipeline where ordering and deduplication semantics matter. `test_context_management.py` verifies that query context is correctly scoped and doesn't leak between calls.

---

## Coverage Requirements & Reports

The minimum threshold is **80%**, enforced in CI via `--cov-fail-under=80`. Coverage is measured against `graphrag_toolkit.lexical_graph` using `.coveragerc` at the module root.

`.coveragerc` excludes `__init__.py` files, `conftest.py`, abstract methods, `__repr__`/`__str__`, `TYPE_CHECKING` blocks, and bare `pass`/`...` stubs. These exclusions are intentional — they represent code that either can't be meaningfully tested in isolation or adds noise to the metric without reflecting real risk.

**Locally:**

```bash
# Run with coverage (matches CI exactly)
PYTHONPATH=src python -m pytest \
  -v --cov-config=.coveragerc --cov=graphrag_toolkit.lexical_graph \
  -l --tb=short --maxfail=1 --cov-fail-under=80 \
  tests/

# Generate XML and HTML reports
PYTHONPATH=src python -m coverage xml
PYTHONPATH=src python -m coverage html

open htmlcov/index.html
```

**In CI:**

The workflow runs against Python 3.10, 3.11, and 3.12. After the test step, `coverage xml` and `coverage html` always run (even on failure). The HTML report is uploaded as the `coverage-report` artifact on every run. If the test step fails on a pull request, a comment is posted to the PR with a link to the artifact.

---

## Maintainer Guide: Adding New Tests

### Process

When adding a new module to `src/`, create a corresponding test file under `tests/unit/<subpath>/test_<module>.py`. If the module lives at `src/graphrag_toolkit/lexical_graph/retrieval/processors/foo.py`, the test file goes at `tests/unit/retrieval/processors/test_foo.py`. New directories need an `__init__.py`.

Group tests into classes by the public method or behaviour being tested, not by the class being tested. A single source class with three distinct public methods should have three test classes.

### What must be included

Every test function needs a docstring stating intent — what condition is being verified and why it matters:

```python
def test_execute_query_handles_connection_error(self, ...):
    # Verify ClientError from Neptune is wrapped in GraphQueryError, not leaked raw.
    ...
```

Any `@pytest.mark.skip` must have a non-empty `reason` that explains why the test is skipped and what needs to happen before it can be re-enabled:

```python
@pytest.mark.skip(reason="batch match result shape is not yet stable — revisit once match() return format is finalised")
```

An empty `reason=""` will fail review.

### What makes a test acceptable for merge

- It doesn't reduce overall coverage below 80%. If you're adding a new module, add enough tests to cover it at the same level as the surrounding code.
- It doesn't make real AWS calls. If your test touches any code path that reaches `GraphRAGConfig`, `boto3.client`, or any AWS SDK method, mock it. The CI environment has no AWS credentials.
- It uses `pytest.importorskip` for any optional dependency (OpenSearch, PGVector, etc.) rather than a try/except around the import at module level — a module-level import failure prevents the entire file from collecting.
- Fixtures are reused from `conftest.py` rather than duplicated inline. If you find yourself writing the same setup in three test files, it belongs in a `conftest.py`.

### Common mistakes

**Patching at the wrong level.** The most common failure is patching `boto3.client` when the code under test accesses AWS through `GraphRAGConfig`. The config layer caches clients, so a low-level patch is often bypassed. Patch the config property or the method that creates the client, not the boto3 entry point.

**Not accounting for `GraphRAGConfig.aws_region`.** Any code path that reads `GraphRAGConfig.aws_region` without a pre-configured region will call `boto3.Session().region_name`, which raises `NoRegionError` in CI. Patch the property directly (see the Config section above).

**Mocking the subject under test.** Some tests in `test_graph_construction.py` mock `constructor.construct` on the object they just instantiated — this tests nothing. Mock dependencies, not the subject.

**Bare `Mock()` without `spec=`.** Use `Mock(spec=VectorIndex)` when the mock stands in for a typed interface. It catches attribute access mistakes at test time rather than silently passing.

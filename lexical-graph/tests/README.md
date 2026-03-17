# Running Tests

```bash
# From the lexical-graph/ directory

# All tests
PYTHONPATH=src python -m pytest tests/

# With coverage (matches CI)
PYTHONPATH=src python -m pytest \
  -v --cov-config=.coveragerc --cov=graphrag_toolkit.lexical_graph \
  -l --tb=short --maxfail=1 --cov-fail-under=50 \
  tests/

# Coverage reports
PYTHONPATH=src python -m coverage xml
PYTHONPATH=src python -m coverage html
open htmlcov/index.html
```

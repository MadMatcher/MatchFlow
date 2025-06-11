# madmatcher-tools

Tools for entity matching and record linkage, providing a flexible and scalable solution for matching and linking records across datasets.

## Features

- Machine learning models for entity matching (both scikit-learn and PySpark)
- Active learning for efficient labeling
- Feature engineering and vectorization
- Data storage and management
- CLI-based labeling interface

## Installation

You can install madmatcher-tools using pip:

```bash
pip install madmatcher-tools
```

For development, install with additional dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

Here's a simple example of using madmatcher-tools:

```python
from madmatcher_tools import SKLearnModel, CLILabeler
from sklearn.ensemble import HistGradientBoostingClassifier

# Create and train a model
model = SKLearnModel(HistGradientBoostingClassifier())
model.train(training_data, vector_col='features', label_column='label')

# Make predictions
predictions = model.predict(test_data, vector_col='features', output_col='prediction')
```

## Documentation

For detailed documentation, visit our [Read the Docs page](https://madmatcher-tools.readthedocs.io/).

## Development

To set up the development environment:

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Build documentation: `cd docs && make html`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

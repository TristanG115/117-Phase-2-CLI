# Model Evaluation System

A comprehensive system for evaluating machine learning models, datasets, and code repositories across multiple quality metrics.

---

## Overview
This system evaluates AI/ML models from various sources (HuggingFace, GitHub) and provides standardized quality scores based on multiple criteria including license compatibility, performance claims, code quality, dataset quality, and more.

---

## Features
- **Multi-source Support**: Evaluates models from HuggingFace, datasets from HuggingFace, and code repositories from GitHub  
- **Comprehensive Metrics**: 8 different quality metrics with weighted scoring  
- **Parallel Processing**: Concurrent evaluation of multiple metrics for performance  
- **Flexible Input**: Process URLs from files with support for grouped evaluations  
- **Logging Support**: Configurable logging levels with file output  
- **Test Suite**: 24 comprehensive test cases with 85%+ code coverage  

---

## Installation

### Prerequisites
- Python **3.7+**
- Internet connection for API access

### Install Dependencies
./run install

This installs:
- requests – For API communications  
- typing-extensions – For enhanced type hints  
- coverage – For test coverage analysis  

---

## Usage

### Basic Usage
./run absolute directory of txt file        # Evaluate URLs from a file  
./run install         # Install dependencies  
./run test            # Run test suite  

---

## URL File Format
Create a text file with URLs, one group per line (comma-separated):

https://github.com/google-research/bert, https://huggingface.co/datasets/bookcorpus/bookcorpus, https://huggingface.co/google-bert/bert-base-uncased  
,,https://huggingface.co/parvk11/audience_classifier_model  
,,https://huggingface.co/openai/whisper-tiny  

---

## Environment Variables

### Logging Configuration
- LOG_LEVEL: Controls logging verbosity  
  - 0 – Silent (no logs)  
  - 1 – INFO level  
  - 2 – DEBUG level  
- LOG_FILE: Path to log file (optional, defaults to console)  

---

## Metrics

The system evaluates models using 8 weighted metrics:

License (20%) – License compatibility (MIT, Apache, BSD, GPL, etc.)  
Performance Claims (15%) – Presence of benchmark results and performance data  
Ramp-up Time (15%) – Documentation quality and ease of use  
Bus Factor (10%) – Number of contributors and project sustainability  
Size Score (10%) – Model size compatibility across hardware platforms  
Dataset & Code Score (10%) – Availability of training data and source code  
Dataset Quality (10%) – Quality metrics for associated datasets  
Code Quality (10%) – Code repository quality and maintenance  

Net Score Calculation:
Net Score = Σ(metric_score × weight) / Σ(weights)  

---

## Output Format
Results are returned in NDJSON format:

{
  "name": "bert-base-uncased",
  "category": "MODEL",
  "net_score": 0.797,
  "net_score_latency": 820,
  "ramp_up_time": 0.5,
  "ramp_up_time_latency": 176,
  "bus_factor": 1.0,
  "bus_factor_latency": 171,
  "performance_claims": 0.5,
  "performance_claims_latency": 103,
  "license": 1.0,
  "license_latency": 193,
  "size_score": {
    "raspberry_pi": 0.0,
    "jetson_nano": 0.3,
    "desktop_pc": 1.0,
    "aws_server": 1.0
  },
  "size_score_latency": 177,
  "dataset_and_code_score": 1.0,
  "dataset_and_code_score_latency": 0,
  "dataset_quality": 1.0,
  "dataset_quality_latency": 0,
  "code_quality": 0.6,
  "code_quality_latency": 216
}

---

## Architecture

### Project Structure
├── run                     # Main entry point script  
├── model_evaluator.py      # Core evaluation orchestrator  
├── url_classifier.py       # URL type classification  
├── handlers/               # Resource-specific handlers  
│   ├── __init__.py  
│   ├── base_resource_handler.py  
│   ├── model_handler.py    # HuggingFace model handling  
│   ├── dataset_handler.py  # HuggingFace dataset handling  
│   └── code_handler.py     # GitHub repository handling  
├── metrics/                # Evaluation metrics  
│   ├── __init__.py  
│   ├── base_metric.py  
│   ├── license_metric.py  
│   ├── size_score_metric.py  
│   ├── ramp_up_time_metric.py  
│   ├── bus_factor_metric.py  
│   ├── performance_claims_metric.py  
│   ├── dataset_and_code_score_metric.py  
│   ├── dataset_quality_metric.py  
│   └── code_quality_metric.py  
├── test_model_evaluator.py # Comprehensive test suite  
└── requirements.txt        # Python dependencies  

---

## Key Components
- ModelEvaluator: Main orchestrator that coordinates evaluation  
- URLClassifier: Identifies URL types (MODEL, DATASET, CODE, UNKNOWN)  
- Resource Handlers: Specialized handlers for each platform/type  
- Metrics: Individual metric calculators with parallel execution  
- Logging System: Configurable logging with file output support  

---

## Testing
Run the comprehensive test suite:
./run test  

Coverage: 85%+ line coverage across core modules  
Test Cases: 24 tests covering:  
- URL classification  
- Resource handler functionality  
- Metric calculations  
- Error handling  
- Logging configuration  
- File processing  

---

## Supported Platforms

### Input Sources
- HuggingFace Models: https://huggingface.co/[org]/[model]  
- HuggingFace Datasets: https://huggingface.co/datasets/[org]/[dataset]  
- GitHub Repositories: https://github.com/[org]/[repo]  

### License Support
- MIT  
- Apache 2.0  
- BSD (2-clause, 3-clause)  
- GPL (v2, v3)  
- LGPL (v2.1, v3)  
- Creative Commons Zero (CC0)  
- Unlicense / Public Domain  

### Hardware Platforms (Size Scoring)
- Raspberry Pi  
- Jetson Nano  
- Desktop PC  
- AWS Server  

---

## Error Handling
The system includes robust error handling for:  
- Invalid URLs  
- Network timeouts  
- API rate limits  
- Invalid file paths  
- Missing dependencies  
- Authentication failures  

---

## License
This project is for educational purposes as part of a software engineering course.

# DevOps-MLOps-Labs Documentation


## Task 1: Prepare the ML Project
1. Fork the repository (or download the ZIP and create a repo).

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117112218.png)

2. Inspect the repo structure and make sure requirements.txt exists.

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117112449.png)

## Task 2: Run the app locally
1. Create a virtualenv, install requirements, and confirm the app runs:

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117113157.png)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117150911.png)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117152733.png)

## Task 3: Write unit tests

## Test Structure

### Test Files Created
```
session2/ml-app/tests/
├── test_data_loader.py
├── test_model.py
├── test_predict.py
└── test_train.py
```

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117153810.png)

### **Data Loader Tests** (`test_data_loader.py`)
**Purpose:** Validate data loading and preprocessing functionality

**Tests Implemented:**
- ✅ **Load Iris Dataset:** Verifies sklearn dataset loads correctly
- ✅ **Data Shape Validation:** Ensures correct number of samples and features
- ✅ **Train-Test Split:** Validates proper data splitting ratios
- ✅ **Data Types:** Confirms numpy arrays are returned
- ✅ **Feature Scaling:** Tests StandardScaler normalization
- ✅ **Missing Values:** Handles edge cases with null data

**Example Test:**
```python
def test_load_data():
    X, y = load_iris_data()
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert isinstance(X, np.ndarray)
```

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117153352.png)

### **Prediction Tests** (`test_predict.py`)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117153115.png)

### **Training Pipeline Tests** (`test_train.py`)
**Purpose:** Validate end-to-end training workflow

**Tests Implemented:**
- ✅ **Full Pipeline Execution:** Tests complete train.py script
- ✅ **Model File Creation:** Verifies models/iris_model.pkl exists
- ✅ **Plot Generation:** Confirms confusion matrix and feature importance plots
- ✅ **Model Persistence:** Ensures saved model can make predictions
- ✅ **Error Handling:** Tests graceful failure on invalid data
- ✅ **Logging Output:** Validates training logs and metrics

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117153729.png)

---

## Task 4: Linting & formatting

1. Add a linter (suggested: flake8) and a minimal config.
(install latest version: flake8>=7.0.0, and update requirements.txt)

2. Ensure flake8 runs and the code meets basic style checks.

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117155109.png)

---

## Continuous Testing Strategy

### **On Every Commit:**
- ✅ Linting checks (flake8)
- ✅ Unit tests
- ✅ Code coverage validation

### **On Pull Requests:**
- ✅ All unit tests
- ✅ Integration tests
- ✅ Coverage report comparison
- ✅ Docker build test

### **Pre-Deployment:**
- ✅ Full test suite
- ✅ Performance benchmarks
- ✅ Security scans
- ✅ Docker image validation

## Key Achievements

✅ **Comprehensive Coverage:** 91% code coverage across all modules  
✅ **Automated Execution:** Tests run automatically on every push/PR  
✅ **Multiple Environments:** Tests pass in local, Docker, and CI environments  
✅ **Quality Gates:** Build fails if tests fail, ensuring code quality  
✅ **Artifact Preservation:** Test results and coverage reports saved for review  
✅ **Fast Feedback:** Full test suite completes in under 15 seconds  
✅ **Maintainable:** Clear test structure following pytest conventions  

## Test Quality Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Code Coverage | 85% | 91% | ✅ Pass |
| Test Count | 15+ | 18 | ✅ Pass |
| Linting Errors | 0 | 0 | ✅ Pass |
| Test Execution Time | <30s | 12s | ✅ Pass |
| Integration Tests | 5+ | 6 | ✅ Pass |

---

## Best Practices Implemented

1. **Test Isolation:** Each test is independent and can run in any order
2. **Fixtures:** Shared setup using pytest fixtures for DRY code
3. **Parameterization:** Multiple test cases with `@pytest.mark.parametrize`
4. **Markers:** Tests categorized for selective execution
5. **Mocking:** External dependencies mocked to avoid side effects
6. **Assertion Messages:** Clear failure messages for debugging
7. **Edge Cases:** Boundary conditions and error scenarios tested
8. **Performance Tests:** Execution time limits enforced

---

## Task 5: GitHub Actions CI workflow
**Objective:** Create a basic GitHub Actions workflow with all required components.

**What We Did:**
- Created `.github/workflows/ci.yml` with two main jobs: `lint-and-test` and `build-docker`
- Implemented checkout, Python setup, dependency installation, linting, and testing
- Added test results artifact upload
- Created Docker build job with image artifact upload
- Fixed duplicate `flake8` entries in `requirements.txt`

**Key Features:**
- Used `actions/setup-python@v5` with pip caching
- Implemented `actions/upload-artifact@v4` for test results
- Built Docker image and saved as tar artifact
- Jobs run sequentially (Docker only builds after tests pass)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117160211.png)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117160245.png)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117155722.png)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117160331.png)

---

## Task 6: Containerise the app
**Objective:** Create production-ready Dockerfile for Task 6.

**What We Did:**
- Used `python:3.10-slim` base image for smaller footprint
- Installed system dependencies (gcc) for scientific libraries
- Implemented layer caching optimization (requirements first)
- Set `PYTHONPATH` environment variable
- Exposed port 8000 for future API integration
- Created necessary directories (`models`, `plots`)
- Set default command to run training

**Dockerfile Features:**
- **Multi-layer optimization:** Requirements copied before code
- **Size reduction:** `--no-cache-dir` flags, apt cache cleanup
- **Flexibility:** Default trains model, but supports other commands
- **Volume-ready:** Directories created for mounting

**Build & Run Commands Provided:**
```bash
# Build
docker build -t iris-classifier:latest .

# Run training
docker run --rm iris-classifier:latest

# With volume mounts
docker run --rm -v $(pwd)/models:/app/models iris-classifier:latest

# Run predictions
docker run --rm -v $(pwd)/models:/app/models iris-classifier:latest python src/predict.py

# Run tests
docker run --rm iris-classifier:latest pytest tests/ -v
```

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117160716.png)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117161712.png)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117161514.png)

![](https://github.com/delat04/DevOps-MLOps-Labs/blob/main/Pasted%20image%2020251117161601.png)

---

## Workflow Optimization
**Objective:** Enhance workflow with best practices and additional features.

**What We Did:**
- Introduced environment variable `WORKING_DIR` for DRY principle
- Added `defaults.run.working-directory` to eliminate repetitive `cd` commands
- Implemented coverage reporting with HTML output
- Added linter output as artifact for review
- Enhanced Docker build with caching and commit SHA tagging
- Added Docker image testing before upload

**Key Improvements:**
- **Caching:** `cache-dependency-path` for faster Python setup
- **Docker cache:** GitHub Actions cache (`type=gha`) for faster builds
- **Tagging:** Images tagged with both `latest` and commit SHA
- **Artifacts:** Linter report, test results, and coverage HTML
- **Validation:** Docker image tested post-build to ensure functionality

---

## Final Deliverables for tasks 5 and 6

### 1. **CI Workflow** (`.github/workflows/ci.yml`)
- ✅ Runs on push and pull requests
- ✅ Checks out code
- ✅ Sets up Python 3.10 with caching
- ✅ Installs dependencies
- ✅ Runs flake8 linter
- ✅ Runs pytest with JUnit XML output
- ✅ Uploads test results as artifacts
- ✅ Builds Docker image
- ✅ Uploads Docker image as artifact

### 2. **Dockerfile** (`session2/ml-app/Dockerfile`)
- ✅ Based on `python:3.10-slim`
- ✅ Installs all dependencies
- ✅ Copies application code
- ✅ Exposes port 8000
- ✅ Runs training by default
- ✅ Supports volume mounts for persistence
- ✅ Optimized for size and caching

### 3. **Updated Requirements** (`requirements.txt`)
- ✅ Fixed duplicate entries
- ✅ Consistent versions across all dependencies

---

## Key Achievements

1. **Automated Testing:** Every push/PR triggers linting and testing
2. **Artifact Management:** Test results and Docker images preserved for review
3. **Docker Integration:** Complete containerization with training capability
4. **Best Practices:** Caching, layer optimization, proper tagging
5. **Flexibility:** Supports multiple run modes (train/predict/test)
6. **Production-Ready:** Workflow and Dockerfile meet enterprise standards

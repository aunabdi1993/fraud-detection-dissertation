# MSc Dissertation Plan: Fraud Detection with Class Imbalance Techniques

**Duration**: 11 months (February - December 2026)  
**Target Completion**: December 2026  
**Institution**: University of London, City - MSc Computer Science

---

## 📊 Executive Summary

### Project Scope

**Focus**: Comprehensive, rigorous comparison of existing class imbalance techniques for financial fraud detection, with production-ready implementation.

**Key Principles:**
- ✅ Excellent execution of existing methods (not novel research)
- ✅ Production-quality system with proper engineering practices
- ✅ Statistical rigor in evaluation
- ✅ Practical deployment considerations

---

## 🎯 Research Focus

### Primary Research Question
> "How do different class imbalance handling techniques compare in performance and practical deployability for financial fraud detection?"

### Secondary Research Questions
1. Which sampling techniques (undersampling, oversampling, hybrid) perform best for highly imbalanced fraud data?
2. How do ensemble methods compare to sampling-based approaches?
3. What are the trade-offs between model performance and inference latency for production deployment?
4. Which techniques provide the best balance of precision, recall, and computational efficiency?

### Evaluation Criteria

**Performance Metrics:**
- PR-AUC (primary metric for imbalanced data)
- ROC-AUC
- F1-Score, Precision, Recall
- Confusion Matrix analysis

**Deployment Metrics:**
- Inference latency (target: <50ms)
- Memory footprint
- Training time
- Model complexity

**Statistical Validation:**
- 5-fold stratified cross-validation
- Paired t-tests for significance
- Friedman test for multiple model comparison

---

## 📅 Timeline Overview

| Phase | Months | Weeks | Key Deliverables |
|-------|--------|-------|------------------|
| Literature Review | Feb-Mar | 1-8 | Research proposal, literature review chapter |
| Data & Baseline | Apr-May | 9-16 | EDA report, feature pipeline, 3 baseline models |
| Model Comparison | Jun-Jul | 17-24 | 10-12 techniques evaluated with statistical tests |
| Production API | Aug-Sep | 25-32 | Deployed FastAPI application with documentation |
| Results & Analysis | Oct-Nov | 33-40 | Complete experiments, visualizations, results chapter |
| Writing & Submission | Dec | 41-44 | Complete 10,000-word dissertation |

---

## 📖 Phase 1: Literature Review (Weeks 1-8)

**Goal**: Establish theoretical foundation and research scope

### Week 1-2: Initial Survey
- [ ] Search IEEE, ACM, arXiv for fraud detection papers
- [ ] Focus areas: fraud detection, class imbalance, production ML
- [ ] Target: 25 key papers covering:
  - Financial fraud detection techniques
  - Class imbalance handling methods
  - Production ML systems
  - Model evaluation for imbalanced data
- [ ] Set up reference management (Zotero/Mendeley)

**Deliverable**: Curated list of 25 papers organized by theme

### Week 3-4: Focused Reading
- [ ] Read 15 foundational papers in detail
- [ ] Document for each paper:
  - Research question and methodology
  - Key techniques and results
  - Datasets used
  - Limitations and gaps identified
- [ ] Create comparison table of techniques
- [ ] Identify most promising approaches

**Deliverable**: Annotated bibliography with technique comparison table

### Week 5-6: Research Proposal
- [ ] Write research proposal (1,500 words) including:
  - Background and motivation
  - Primary and secondary research questions
  - Proposed methodology overview
  - Expected contributions
  - Timeline and milestones
- [ ] Submit for supervisor approval
- [ ] Revise based on feedback

**Deliverable**: Approved research proposal

### Week 7-8: Complete Literature Review Chapter
- [ ] Read remaining 10 papers
- [ ] Write Literature Review chapter (2,000 words):
  - Section 1: Financial fraud detection overview
  - Section 2: Class imbalance techniques taxonomy
  - Section 3: Production ML considerations
  - Section 4: Gap analysis and research positioning
- [ ] Create summary tables and conceptual diagrams
- [ ] Submit draft for feedback

**Deliverable**: Literature Review chapter draft (2,000 words)

### Phase 1 Technical Outputs
- ✅ 25 papers reviewed and categorized
- ✅ Technique comparison framework established
- ✅ Research scope clearly defined
- ✅ Literature review chapter completed

---

## 💾 Phase 2: Data & Baseline (Weeks 9-16)

**Goal**: Establish data pipeline and baseline model performance

### Week 9-10: Data Acquisition & Setup
- [ ] Download Kaggle Credit Card Fraud dataset
- [ ] Set up project repository structure
- [ ] Configure Python environment (requirements.txt)
- [ ] Set up data versioning (DVC or git-lfs)
- [ ] Document dataset characteristics:
  - Size: 284,807 transactions
  - Features: 28 PCA features + Time + Amount
  - Class distribution: 0.172% fraud (492/284,807)
  - Temporal range and patterns
- [ ] Create data loading module (`src/data/loader.py`)

**Deliverable**: Repository with documented dataset

### Week 11-12: Exploratory Data Analysis
- [ ] Analyze class distribution and imbalance ratio
- [ ] Study feature distributions by class
- [ ] Compute correlation matrix
- [ ] Analyze temporal patterns
- [ ] Create visualization suite (10-15 plots):
  - Class balance visualization
  - Feature distributions (histograms, KDE)
  - Correlation heatmap
  - Temporal transaction patterns
  - Amount distribution by class
  - Box plots for outlier detection
- [ ] Identify data quality issues
- [ ] Document findings in `notebooks/01_eda.ipynb`

**Deliverable**: EDA notebook with documented insights

### Week 13-14: Feature Engineering
- [ ] Implement RFM features (`src/features/rfm.py`):
  - Recency: Time since last transaction
  - Frequency: Transaction count in time window
  - Monetary: Average/sum transaction amounts
- [ ] Implement velocity features (`src/features/velocity.py`):
  - Transaction rate (transactions/hour)
  - Amount velocity (spending rate)
  - Acceleration features
- [ ] Implement aggregation features (`src/features/aggregation.py`):
  - Rolling statistics (mean, std, min, max)
  - Historical patterns
  - Time-based aggregations
- [ ] Create feature engineering pipeline
- [ ] Document in `notebooks/02_feature_engineering.ipynb`

**Deliverable**: Feature engineering pipeline with unit tests

### Week 15-16: Baseline Models
- [ ] Establish data splits:
  - Training: 60% (stratified)
  - Validation: 20% (stratified)
  - Test: 20% (stratified, held out)
- [ ] Implement three baseline models:
  - **Logistic Regression** (`src/models/baseline.py`)
  - **Decision Tree**
  - **Random Forest**
- [ ] Train on imbalanced data (no resampling)
- [ ] Evaluate with multiple metrics:
  - PR-AUC (primary)
  - ROC-AUC
  - F1-Score, Precision, Recall
  - Confusion matrices
- [ ] Set up MLflow experiment tracking
- [ ] Document baseline performance
- [ ] Create results comparison table

**Deliverable**: Three baseline models with performance benchmarks

### Phase 2 Technical Outputs
- ✅ Clean, versioned dataset
- ✅ Comprehensive EDA with visualizations
- ✅ Feature engineering pipeline (RFM, velocity, aggregations)
- ✅ Proper train/val/test splits
- ✅ Three baseline models evaluated
- ✅ MLflow tracking configured

---

## 🔬 Phase 3: Model Comparison (Weeks 17-24)

**Goal**: Comprehensive evaluation of class imbalance techniques

### Week 17-18: Sampling Techniques
- [ ] Implement sampling methods (`src/models/sampling.py`):
  - Random Undersampling
  - Random Oversampling
  - SMOTE (Synthetic Minority Over-sampling)
  - SMOTE + Tomek Links
  - ADASYN (Adaptive Synthetic Sampling)
- [ ] Apply each technique to 3 base models:
  - Logistic Regression
  - Random Forest  
  - XGBoost
- [ ] Total experiments: 5 sampling × 3 models = 15 variants
- [ ] Track all metrics in MLflow
- [ ] Measure computational cost per technique
- [ ] Document in `notebooks/04_sampling_comparison.ipynb`

**Deliverable**: 15 model variants with performance comparison

### Week 19-20: Ensemble Methods
- [ ] Implement ensemble approaches (`src/models/ensemble.py`):
  - Balanced Random Forest
  - RUSBoost (Random Under-Sampling Boost)
  - EasyEnsemble
  - XGBoost with `scale_pos_weight`
  - LightGBM with `is_unbalance`
- [ ] Compare against best sampling technique results
- [ ] Profile inference latency for each model
- [ ] Measure memory footprint
- [ ] Create latency vs. performance scatter plot

**Deliverable**: 5 ensemble models with performance/latency analysis

### Week 21-22: Cost-Sensitive Learning
- [ ] Implement cost-sensitive approaches:
  - Class weight optimization
  - Custom threshold tuning
  - (Optional) Focal Loss if time permits
- [ ] Grid search for optimal class weights
- [ ] Threshold optimization on validation set
- [ ] Compare against ensemble methods
- [ ] Create comprehensive comparison table
- [ ] Identify top 3-5 candidates for production

**Deliverable**: Cost-sensitive models and master comparison table

### Week 23-24: Hyperparameter Tuning
- [ ] Select top 3 models from all experiments
- [ ] Configure Optuna for hyperparameter optimization:
  - Define search spaces per model
  - Set objective: maximize PR-AUC
  - Run 50-100 trials per model
  - Use validation set for optimization
- [ ] Train final models with optimal hyperparameters
- [ ] Perform 5-fold cross-validation
- [ ] Final evaluation on held-out test set
- [ ] Statistical significance testing:
  - Paired t-tests between models
  - Friedman test for overall comparison
- [ ] Document optimal configurations
- [ ] Save best models to `models/production/`

**Deliverable**: Top 3 optimized models ready for production

### Phase 3 Technical Outputs
- ✅ 10-12 different approaches implemented and evaluated
- ✅ Comprehensive performance comparison table
- ✅ Statistical significance tests
- ✅ Latency and resource usage analysis
- ✅ Top 3 production candidates identified
- ✅ All experiments logged in MLflow

---

## 🚀 Phase 4: Production API (Weeks 25-32)

**Goal**: Deploy production-ready fraud detection API

### Week 25-26: API Development
- [ ] Set up FastAPI project structure (`src/api/`)
- [ ] Implement core endpoints:
  - `POST /predict` - Single transaction prediction
  - `POST /batch-predict` - Batch predictions
  - `GET /health` - Health check endpoint
  - `GET /model-info` - Model metadata and version
- [ ] Create Pydantic schemas (`src/api/models.py`):
  - TransactionRequest
  - TransactionResponse
  - BatchPredictionRequest
  - BatchPredictionResponse
  - ModelInfo
- [ ] Implement input validation
- [ ] Add error handling and logging
- [ ] Configure CORS for web access
- [ ] Set up API versioning (v1)

**Deliverable**: Functional FastAPI application with all endpoints

### Week 27-28: Model Integration
- [ ] Create inference module (`src/api/inference.py`)
- [ ] Load best model from experiments
- [ ] Implement feature engineering pipeline for inference:
  - Load preprocessing transformers
  - Apply feature calculations
  - Handle missing/invalid values
- [ ] Optimize inference code:
  - Implement model caching
  - Batch processing for multiple predictions
  - Input vectorization
  - Warm-up on startup
- [ ] Profile and optimize:
  - Target: <50ms latency for single prediction
  - Target: <100ms for batch of 10 predictions
- [ ] Benchmark performance with sample data

**Deliverable**: Optimized inference pipeline integrated with API

### Week 29-30: Documentation & Testing
- [ ] Write comprehensive API documentation:
  - OpenAPI/Swagger auto-generated docs
  - README with usage examples
  - Request/response examples
  - Error codes and handling
- [ ] Implement test suite (`tests/test_api.py`):
  - Unit tests for each endpoint
  - Test edge cases (invalid input, missing fields)
  - Test error handling
  - Test batch processing
- [ ] Integration tests:
  - End-to-end prediction flow
  - Model loading and caching
  - Concurrent request handling
- [ ] Load testing:
  - Use locust or hey for load testing
  - Test with 100 concurrent users
  - Identify bottlenecks
- [ ] Achieve >80% code coverage

**Deliverable**: Fully tested API with comprehensive documentation

### Week 31-32: Deployment
- [ ] Create Docker configuration:
  - Write `Dockerfile` for API
  - Create `docker-compose.yml` for local testing
  - Optimize image size (multi-stage build)
  - Include health checks
- [ ] Deploy to cloud platform:
  - Railway.app (recommended - free tier)
  - Alternative: Render.com or Fly.io
- [ ] Configure deployment:
  - Environment variables
  - Secret management
  - Health check endpoints
  - Auto-restart on failure
- [ ] Test deployed API:
  - Verify all endpoints work
  - Test latency from external network
  - Verify error handling
  - Load test deployed version
- [ ] Document deployment process:
  - Deployment guide (`docs/deployment_guide.md`)
  - Environment setup instructions
  - Rollback procedures

**Deliverable**: Live, deployed API with public URL and documentation

### Phase 4 Technical Outputs
- ✅ Working FastAPI application
- ✅ All endpoints functional with validation
- ✅ Comprehensive API documentation (Swagger UI)
- ✅ Unit tests passing (>80% coverage)
- ✅ Deployed to cloud platform
- ✅ Performance benchmarks meeting targets
- ✅ Deployment documentation complete

---

## 📊 Phase 5: Results & Analysis (Weeks 33-40)

**Goal**: Complete comprehensive experimental evaluation

### Week 33-34: Statistical Validation
- [ ] Re-run all models with fixed random seeds
- [ ] Perform 5-fold stratified cross-validation:
  - Run for each of the 10-12 approaches
  - Calculate mean and std dev for all metrics
  - Generate fold-wise results
- [ ] Statistical significance testing:
  - Paired t-tests between top models
  - Friedman test for multiple model comparison
  - Post-hoc Nemenyi test if needed
  - Bonferroni correction for multiple comparisons
- [ ] Generate comprehensive results tables:
  - Performance metrics table (PR-AUC, ROC-AUC, F1, etc.)
  - Statistical significance matrix
  - Computational cost comparison (training time, inference time)
  - Resource usage table (memory, CPU)
- [ ] Document experimental configurations for reproducibility

**Deliverable**: Statistically validated experimental results

### Week 35-36: Visualization Suite
- [ ] Create publication-quality figures:
  - **Performance plots:**
    - PR curves for all models (with confidence intervals)
    - ROC curves comparison
    - Bar charts for metric comparison
  - **Model analysis:**
    - Confusion matrices (top 5 models)
    - Feature importance plots (top 3 models)
    - Learning curves
  - **Deployment analysis:**
    - Performance vs. latency scatter plot
    - Training time comparison bar chart
    - Memory footprint comparison
- [ ] Error analysis:
  - False positive analysis (common characteristics)
  - False negative analysis (missed fraud patterns)
  - Create failure case studies (5-10 examples)
  - Visualize error distributions
- [ ] Create results dashboard (optional, interactive with Plotly)
- [ ] Save all figures in `results/figures/` (PDF and PNG)

**Deliverable**: Complete visualization suite for dissertation

### Week 37-38: Deployment Analysis
- [ ] Inference latency profiling:
  - Measure 1000 predictions per model
  - Calculate latency statistics:
    - Mean, median, std dev
    - p95 and p99 percentiles
    - Min and max latency
  - Create latency distribution plots
- [ ] Memory footprint analysis:
  - Measure RAM usage during inference
  - Model file size comparison
  - Loading time measurement
- [ ] Accuracy vs. latency trade-off analysis:
  - Create scatter plot (PR-AUC vs. latency)
  - Identify Pareto frontier
  - Highlight models meeting <50ms target
- [ ] Production deployment considerations:
  - Analyze which models meet latency requirements
  - Estimate resource requirements for scaling
  - Document deployment trade-offs
- [ ] Create deployment recommendations table

**Deliverable**: Deployment feasibility analysis

### Week 39-40: Results Chapter
- [ ] Write Results & Evaluation chapter (2,500 words):
  - **Section 1**: Baseline Results
    - Present baseline model performance
    - Analyze performance on imbalanced data
  - **Section 2**: Sampling Technique Comparison
    - Compare all sampling approaches
    - Statistical significance analysis
  - **Section 3**: Ensemble Methods Performance
    - Ensemble vs. sampling comparison
    - Best performing approach identification
  - **Section 4**: Hyperparameter Optimization
    - Present tuned model results
    - Improvement analysis
  - **Section 5**: Latency and Deployment Analysis
    - Performance-latency trade-offs
    - Production deployment recommendations
  - **Section 6**: Statistical Analysis Summary
    - Significance tests summary
    - Overall best model selection
- [ ] Create all tables with captions
- [ ] Insert all figures with references
- [ ] Cross-reference throughout text
- [ ] Submit draft for supervisor review

**Deliverable**: Complete Results & Evaluation chapter (2,500 words)

### Phase 5 Technical Outputs
- ✅ All experiments completed with statistical rigor
- ✅ Publication-quality figures and tables
- ✅ Comprehensive error analysis
- ✅ Deployment feasibility study
- ✅ Results chapter written
- ✅ All data backed up and version controlled

---

## ✍️ Phase 6: Writing & Submission (Weeks 41-44)

**Goal**: Complete and submit dissertation

### Week 41-42: Core Chapters
- [ ] **Introduction Chapter** (1,500 words):
  - Background on financial fraud detection
  - Problem statement and motivation
  - Research questions (primary + secondary)
  - Contributions summary
  - Thesis structure overview
  
- [ ] **Methodology Chapter** (2,000 words):
  - Dataset description and justification
  - Data preprocessing approach
  - Feature engineering methodology
  - Experimental design and setup
  - Evaluation metrics and rationale
  - Statistical testing methodology
  - Models and techniques compared

- [ ] **System Architecture Chapter** (1,500 words):
  - Requirements analysis
  - System design and architecture
  - API design decisions
  - Technology stack justification
  - Implementation highlights
  - Deployment strategy

- [ ] **Discussion Chapter** (1,000 words):
  - Interpretation of results
  - Comparison with literature findings
  - Practical implications for practitioners
  - Limitations and threats to validity
  - Lessons learned
  - Unexpected findings

- [ ] **Conclusion Chapter** (1,000 words):
  - Summary of key findings
  - Research questions answered
  - Contributions recap
  - Future work and extensions
  - Final reflections

**Deliverable**: Complete first draft of all chapters (~8,000 words)

### Week 43: Polish & References
- [ ] Proofread entire dissertation
- [ ] Complete bibliography:
  - Export from reference manager
  - Verify all in-text citations present
  - Check formatting consistency
  - Ensure 25+ references
- [ ] Create appendices:
  - **Appendix A**: Experimental configurations
  - **Appendix B**: Additional results tables
  - **Appendix C**: API documentation summary
  - **Appendix D**: Code repository structure
- [ ] Format all figures and tables:
  - Number sequentially
  - Add descriptive captions
  - Ensure all are referenced in text
  - Check image quality/resolution
- [ ] Write abstract (300 words):
  - Background (2 sentences)
  - Methods (3 sentences)
  - Results (3 sentences)
  - Conclusions (2 sentences)
- [ ] Generate table of contents
- [ ] Write acknowledgments
- [ ] Add list of figures and tables

**Deliverable**: Polished dissertation ready for review

### Week 44: Final Review & Submission
- [ ] Submit draft to supervisor for feedback
- [ ] Implement supervisor feedback:
  - Address technical comments
  - Clarify ambiguous sections
  - Add missing references
  - Fix formatting issues
- [ ] Final comprehensive proofread:
  - Read entire document aloud for flow
  - Check grammar and spelling
  - Verify technical accuracy
  - Ensure consistency in terminology
- [ ] Verify university formatting requirements:
  - Margins (typically 2.5cm)
  - Font (typically Times New Roman 12pt)
  - Line spacing (typically 1.5 or double)
  - Page numbers
  - Header/footer requirements
  - Cover page format
  - Declaration of authorship page
- [ ] Generate final PDF
- [ ] Final quality checks:
  - All figures render correctly
  - All tables formatted properly
  - All links work (if digital submission)
  - File size acceptable
- [ ] Submit through university portal
- [ ] Archive all project materials

**Deliverable**: Submitted dissertation (10,000 words)

### Phase 6 Technical Outputs
- ✅ Complete dissertation (10,000 words)
- ✅ All chapters written and integrated
- ✅ Bibliography complete (~25+ references)
- ✅ All figures and tables properly formatted
- ✅ Supervisor feedback incorporated
- ✅ University formatting requirements met
- ✅ Successfully submitted

---

## 📚 Dissertation Structure (Target: 10,000 words)

### Chapter Breakdown

| Chapter | Word Count | Key Content |
|---------|-----------|-------------|
| 1. Introduction | 1,500 | Background, problem, research questions, contributions |
| 2. Literature Review | 2,000 | Fraud detection, class imbalance, production ML, gaps |
| 3. Methodology | 2,000 | Dataset, preprocessing, experimental design, metrics |
| 4. System Architecture | 1,500 | Requirements, API design, deployment, tech stack |
| 5. Results & Evaluation | 2,500 | All experimental results, statistical tests, analysis |
| 6. Discussion | 1,000 | Interpretation, implications, limitations |
| 7. Conclusion | 1,000 | Summary, answers to RQs, future work |
| Abstract | 300 | Project summary |
| References | - | Bibliography (~25 papers) |
| Appendices | - | Additional tables, configs, code samples |
| **TOTAL** | **~10,000** | Complete dissertation |

---

## 🎯 Project Scope Definition

### In Scope (Core Deliverables)
- ✅ Comprehensive comparison of 10-12 class imbalance techniques
- ✅ Rigorous experimental methodology (5-fold CV, statistical tests)
- ✅ Production-quality FastAPI implementation
- ✅ Deployed fraud detection service
- ✅ 10,000-word dissertation with proper academic rigor
- ✅ Complete documentation and reproducible code

### Out of Scope (Future Work)
- ❌ Novel algorithm development (focus on existing techniques)
- ❌ Multiple datasets (single dataset, done thoroughly)
- ❌ Full MLOps platform (monitoring, A/B testing, drift detection)
- ❌ Extensive literature review (25 papers, not 50+)

This scoping enables deep, high-quality work on a focused problem rather than shallow coverage of many topics.

---

## 🚀 Project Initialization Steps

### Week 1: Setup & Foundation

**Repository Setup:**
- [ ] Create GitHub repository: `fraud-detection-msc`
- [ ] Initialize with README.md and .gitignore
- [ ] Set up project structure (see README.md)
- [ ] Configure branch protection (main branch)

**Development Environment:**
- [ ] Create Python virtual environment (3.9+)
- [ ] Install base dependencies (requirements.txt)
- [ ] Set up Jupyter notebook environment
- [ ] Configure VS Code/PyCharm for project

**Reference Management:**
- [ ] Set up Zotero or Mendeley
- [ ] Create bibliography structure
- [ ] Import initial papers

**Data Acquisition:**
- [ ] Create Kaggle account (if needed)
- [ ] Download Credit Card Fraud Detection dataset
- [ ] Verify data integrity (checksums)
- [ ] Document data characteristics

**Experiment Tracking:**
- [ ] Install and configure MLflow
- [ ] Set up experiment tracking structure
- [ ] Test logging functionality

**Version Control:**
- [ ] Set up DVC or git-lfs for data versioning
- [ ] Create .gitignore for data files and models
- [ ] Make initial commit with project structure

**Documentation:**
- [ ] Create DISSERTATION_PLAN.md (this file)
- [ ] Create README.md with project overview
- [ ] Set up docs/ folder structure
- [ ] Create TODO.md for tracking progress

---

## 🛠️ Technical Implementation Notes

### Code Quality Standards
- **Testing**: Maintain >80% code coverage
- **Documentation**: Docstrings for all functions (Google style)
- **Type Hints**: Use Python type hints throughout
- **Linting**: Follow PEP 8 (use black, flake8)
- **Version Control**: Commit frequently with descriptive messages

### Reproducibility Requirements
- **Random Seeds**: Fix all random seeds (42 by convention)
- **Environment**: Document exact package versions
- **Data Versioning**: Track data splits and preprocessing
- **Configuration**: Use YAML files for experiment configs
- **Logging**: Comprehensive logging of all experiments

### Performance Optimization
- **Vectorization**: Use NumPy/Pandas vectorized operations
- **Caching**: Cache expensive computations
- **Profiling**: Profile code to identify bottlenecks
- **Memory**: Monitor memory usage for large datasets
- **Batching**: Use batch processing for inference

### Best Practices for Claude Code Collaboration
When working with Claude Code on this project:

1. **Give Context**: Start each session by referencing this DISSERTATION_PLAN.md
2. **Specify File Paths**: Always provide exact file paths (e.g., `src/models/sampling.py`)
3. **Request Tests**: Ask for tests alongside implementation
4. **Incremental Development**: Build features incrementally, test as you go
5. **Documentation First**: Have Claude Code create docstrings and comments
6. **Review Output**: Always review and understand generated code
7. **Commit Regularly**: Commit working code before major changes

### Useful Commands for Claude Code
```bash
# Create a new module
claude "Create src/features/rfm.py with functions for recency, frequency, and monetary features. Include docstrings and type hints."

# Add tests
claude "Create tests/test_features.py with unit tests for the RFM features module."

# Debug issues
claude "I'm getting an error in src/models/baseline.py line 45. Help me debug."

# Optimize code
claude "Review src/api/inference.py and suggest performance optimizations."

# Generate documentation
claude "Create API documentation in docs/api_documentation.md based on src/api/main.py"
```

---

## 📚 Recommended Resources

### Academic Writing
- *How to Write a Thesis* by Umberto Eco
- *Writing for Computer Science* by Justin Zobel
- University writing guidelines

### Machine Learning
- *Imbalanced Learning* by He & Garcia (2009)
- *scikit-learn* documentation on imbalanced-learn
- MLOps best practices

### Production ML
- *Designing Machine Learning Systems* by Chip Huyen
- FastAPI official documentation
- Docker best practices

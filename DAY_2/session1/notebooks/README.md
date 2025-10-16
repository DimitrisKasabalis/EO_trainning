# DAY 2 - Session 1: Random Forest Theory Notebooks

## Overview

This directory contains comprehensive Jupyter notebooks for teaching Random Forest fundamentals for Earth Observation applications. The notebooks are designed for the **CopPhil 4-Day Advanced Online Training on AI/ML for Earth Observation**.

## Files

### 1. `session1_theory_notebook_STUDENT.ipynb`
**Target Audience**: Training participants
**Purpose**: Interactive learning with exercises

**Features**:
- Learning objectives and motivation
- Step-by-step explanations with visualizations
- Interactive exercises with TODO markers
- Self-assessment quiz questions
- Space for students to write answers
- Estimated time: 70 minutes

**Content Sections**:
- A. Introduction and Setup (5 min)
- B. Decision Trees Interactive Demo (15 min)
- C. Random Forest Voting Mechanism (15 min)
- D. Feature Importance Analysis (10 min)
- E. Confusion Matrix Interpretation (15 min)
- F. Concept Check Quiz (10 min)

### 2. `session1_theory_notebook_INSTRUCTOR.ipynb`
**Target Audience**: Training instructors/facilitators
**Purpose**: Complete reference with solutions and teaching notes

**Additional Features**:
- Complete solutions to all exercises
- Detailed explanations for quiz questions
- Teaching tips and discussion points
- Timing guidance for each section
- Common student questions with answers
- Troubleshooting advice
- Extension activities for advanced students

## Learning Objectives

By completing these notebooks, participants will be able to:

1. **Understand Decision Trees**: Explain how recursive splitting creates decision rules
2. **Grasp Ensemble Learning**: Describe bootstrap sampling, random feature selection, and voting
3. **Interpret Feature Importance**: Analyze which spectral bands contribute most to classification
4. **Evaluate Model Performance**: Read confusion matrices and calculate precision/recall
5. **Apply to EO Context**: Connect concepts to satellite image classification

## Prerequisites

### Python Libraries
```python
numpy >= 1.19.0
pandas >= 1.1.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
```

### Knowledge Prerequisites
- Basic Python programming
- Understanding of satellite imagery concepts (spectral bands, reflectance)
- Familiarity with classification problems
- Basic statistics (mean, variance)

## Running the Notebooks

### Option 1: Google Colab (Recommended)
1. Upload notebook to Google Colab
2. All required libraries are pre-installed
3. Run cells sequentially
4. Modify exercises and experiment

### Option 2: Local Jupyter
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch Jupyter
jupyter notebook

# Open the student notebook
```

### Option 3: VS Code
1. Install Python extension
2. Install Jupyter extension
3. Open .ipynb file
4. Select Python kernel
5. Run cells interactively

## Teaching Guide

### Session Structure

**Recommended Flow**:
1. **Introduction** (5 min): Learning objectives and motivation
2. **Core Concepts** (35 min): Sections B, C, D with live demonstrations
3. **Evaluation** (15 min): Section E on confusion matrices
4. **Assessment** (10 min): Section F quiz
5. **Q&A** (5 min): Address questions and preview hands-on session

**Total**: 70 minutes theory

### Teaching Tips

#### Section B: Decision Trees
- **Demo Strategy**: Run a single tree with max_depth=1, then gradually increase
- **Visualization Focus**: Point out rectangular boundaries
- **Interactive Element**: Have students predict what max_depth=10 will look like
- **Key Insight**: Trees memorize with unlimited depth

#### Section C: Random Forest
- **Analogy**: Use "wisdom of the crowd" or "committee of experts"
- **Visualization Focus**: Show how ensemble smooths individual tree errors
- **Interactive Element**: Poll class on how many trees they think is optimal
- **Key Insight**: More trees = more stable, diminishing returns

#### Section D: Feature Importance
- **Context**: Connect to Sentinel-2 bands they'll use in hands-on
- **Caution**: Emphasize importance ≠ causation
- **Interactive Element**: Ask students to predict which bands will be important before running
- **Key Insight**: Derived indices often rank high

#### Section E: Confusion Matrix
- **Real-World Context**: Use disaster mapping example (precision vs. recall trade-off)
- **Common Confusion**: Explain difference between user's and producer's accuracy
- **Interactive Element**: Have students identify confused classes in the matrix
- **Key Insight**: Overall accuracy can be misleading

#### Section F: Quiz
- **Format Options**:
  - Individual → pair discussion → class discussion
  - Kahoot/Mentimeter for real-time polling
  - Think-pair-share for deeper engagement
- **Discussion**: Spend more time on questions students get wrong
- **Key Insight**: Reinforce core concepts before hands-on

### Common Student Questions

**Q: "Why use Random Forest instead of a single decision tree?"**
A: Single tree is unstable (high variance). Small data changes → completely different tree. RF averages many trees → stable predictions. Show Section C visualization.

**Q: "How many trees should I use in practice?"**
A: Typically 100-500. More trees never hurts accuracy (just computation time). Show Section C exercise demonstrating diminishing returns. Rule of thumb: Use enough until validation error stabilizes.

**Q: "What if my classes are imbalanced (e.g., 90% forest, 10% mangrove)?"**
A: Three options:
1. Use `class_weight='balanced'` parameter (automatic adjustment)
2. Oversample minority class (duplicate samples)
3. Undersample majority class (remove samples)
4. Adjust decision threshold for rare class

**Q: "Why is NDVI more important than NIR, when NDVI is derived from NIR?"**
A: NDVI combines NIR and Red in a normalized way that enhances vegetation contrast. It's a better discriminator than NIR alone. But importance is shared among correlated features.

**Q: "My Urban class has low precision. What should I do?"**
A: Urban is heterogeneous (buildings, roads, bare soil). Try:
1. Collect more diverse urban training samples
2. Split into sub-classes (residential, commercial, industrial)
3. Add texture features (urban is spatially complex)
4. Use temporal features (urban doesn't change seasonally)

**Q: "Should I always maximize accuracy?"**
A: No! Depends on application:
- Disaster mapping → maximize recall (don't miss affected areas)
- Urban taxation → maximize precision (avoid false accusations)
- General mapping → balance with F1-score
Ask: "What's more costly, false positives or false negatives?"

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named sklearn` | Install: `pip install scikit-learn` |
| Plots not showing in Jupyter | Add: `%matplotlib inline` at top |
| Notebook runs slowly | Reduce `n_samples` or mesh grid resolution |
| Different results each run | Ensure `random_state=42` is set |
| Memory error | Reduce dataset size or use fewer trees |

## Key Concepts Summary

### Decision Trees
- Recursive partitioning of feature space
- Greedy algorithm (locally optimal splits)
- Axis-aligned boundaries
- Prone to overfitting if deep

### Random Forest
- **Bootstrap sampling**: Each tree trains on random subset (with replacement)
- **Random feature selection**: Each split considers random subset of features
- **Voting**: Final prediction by majority vote (classification) or average (regression)
- **Benefits**: Reduced variance, more stable, handles high dimensions

### Feature Importance
- Based on mean decrease in impurity (Gini or entropy)
- Averaged across all trees
- Useful for interpretation and feature selection
- **Caution**: Correlated features share importance

### Evaluation Metrics
- **Accuracy**: Overall correctness (can be misleading with imbalanced data)
- **Precision**: TP / (TP + FP) - reliability of positive predictions
- **Recall**: TP / (TP + FN) - completeness of detection
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows which classes are confused

## Pedagogical Design Principles

### 1. Concrete to Abstract
- Start with visual 2D examples
- Progress to multi-dimensional EO data
- Build intuition before formulas

### 2. Active Learning
- Interactive exercises with TODO markers
- Hands-on parameter exploration
- Self-assessment quiz

### 3. EO Contextualization
- Use EO terminology (spectral bands, land cover)
- Connect to Philippine applications
- Real-world examples (disaster mapping, etc.)

### 4. Multi-Modal Explanations
- Visual (plots, decision boundaries)
- Textual (markdown explanations)
- Mathematical (formulas where appropriate)
- Analogical (expert committee, wisdom of crowds)

### 5. Scaffolded Complexity
- Simple binary classification → multi-class
- Few features → many features
- Single tree → ensemble
- Training accuracy → validation metrics

## Expected Learning Outcomes

After completing these notebooks, participants should be able to:

### Knowledge (Understand)
- Explain how decision trees make splits
- Describe bootstrap sampling and random feature selection
- Define precision, recall, and F1-score

### Skills (Apply)
- Train a Random Forest classifier in scikit-learn
- Interpret feature importance plots
- Read and analyze confusion matrices
- Choose appropriate evaluation metrics

### Attitudes (Appreciate)
- Value ensemble methods over single models
- Recognize importance of evaluation beyond overall accuracy
- Understand data quality matters more than algorithm complexity

## Connection to Hands-On Session

These theory notebooks prepare participants for the **Session 1 Hands-On** where they will:

1. Load actual Sentinel-2 imagery (Palawan, Philippines)
2. Extract training samples from land cover reference data
3. Train Random Forest on real multi-spectral data
4. Generate wall-to-wall land cover classification
5. Validate results and interpret errors
6. Optimize hyperparameters

**Key Bridge**: The concepts learned here (feature importance, confusion analysis, parameter tuning) directly apply to the real-world classification task.

## Customization Guide

### For Different Audiences

**Beginner Programmers**:
- Add more code comments
- Include Python basics refresher
- Provide more scaffolded exercises

**Advanced Practitioners**:
- Add extension activities (cross-validation, hyperparameter tuning)
- Include comparison with other algorithms (XGBoost, SVM)
- Discuss advanced topics (permutation importance, SHAP values)

**Non-Philippine Context**:
- Replace land cover types with local classes
- Use different geographic examples
- Adjust spectral signatures to local environment

### For Different Time Constraints

**Short Version (45 min)**:
- Skip Section B exercise (just demo)
- Reduce Section E depth
- Make quiz optional/homework

**Extended Version (90 min)**:
- Add live coding exercises
- Include group discussions
- Add hands-on debugging practice

## Additional Resources

### Further Reading
- Breiman (2001): Original Random Forests paper
- Belgiu & Drăguţ (2016): RF in remote sensing review
- Scikit-learn documentation: User guide and tutorials

### Online Tutorials
- Scikit-learn Random Forest tutorial
- Google Earth Engine classification guides
- Philippine EO community resources (PhilSA, DOST-ASTI)

### Tools
- Google Colab: Free GPU/TPU access
- Kaggle Notebooks: Pre-configured environment
- Jupyter Lab: Enhanced notebook interface

## Feedback and Improvements

This is a living educational resource. Suggested improvements:

### Short-term
- Add interactive widgets (ipywidgets) for parameter exploration
- Include audio narration for key concepts
- Create video walkthroughs for complex sections

### Long-term
- Develop auto-graded exercises
- Create complementary video lectures
- Build interactive web dashboard for exploration

## License and Attribution

**Developed for**: CopPhil 4-Day Advanced Online Training on AI/ML for Earth Observation
**Target Audience**: Philippine EO professionals
**Version**: 1.0
**Last Updated**: 2025-01-XX

## Contact

For questions, suggestions, or bug reports:
- Open an issue in the course repository
- Contact course instructors
- Join the CopPhil Digital Space Campus community

---

**Ready to start learning? Open the STUDENT notebook and begin your Random Forest journey!**

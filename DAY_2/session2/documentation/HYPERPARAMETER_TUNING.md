# Random Forest Hyperparameter Tuning Guide
**Session 2: Advanced Palawan Land Cover Classification**

---

## Overview

Random Forest classifiers have several hyperparameters that affect performance. This guide provides practical recommendations for tuning these parameters in Google Earth Engine.

---

## Key Hyperparameters

### 1. Number of Trees (`numberOfTrees`)

**What it does:**  
Controls how many decision trees are in the ensemble.

**Range:** 10-1000 (typical: 100-500)

**Effects:**
- **More trees:**
  - ✅ Generally better accuracy (diminishing returns after ~200)
  - ✅ More stable predictions
  - ✅ Better out-of-bag error estimates
  - ❌ Slower training and prediction
  - ❌ Higher memory usage

**Recommendations:**
- **Exploratory analysis:** 50-100 trees (fast iterations)
- **Production:** 200-500 trees (optimal balance)
- **Small datasets:** 100 trees usually sufficient
- **Large datasets:** 200+ trees for stability

**Example:**
```python
classifier = ee.Classifier.smileRandomForest(numberOfTrees=200)
```

**Tuning Results (Palawan case study):**

| Trees | Accuracy | Training Time | Notes |
|-------|----------|---------------|-------|
| 50    | 84.2%    | Fast          | Good for testing |
| 100   | 86.5%    | Moderate      | Good balance |
| 200   | 87.3%    | Slow          | ⭐ Recommended |
| 500   | 87.5%    | Very slow     | Marginal improvement |

---

### 2. Variables Per Split (`variablesPerSplit`)

**What it does:**  
Number of features randomly selected at each split.

**Options:**
- `null` or `None`: Default (√n where n = total features)
- Integer: Specific number
- For 23 features: √23 ≈ 5

**Effects:**
- **More variables:**
  - ✅ Trees can find best splits
  - ❌ Trees become more similar (less diversity)
  - ❌ Higher correlation between trees

- **Fewer variables:**
  - ✅ More tree diversity
  - ✅ Reduce overfitting
  - ❌ May miss optimal splits

**Recommendations:**
- **Default (√n):** Best for most cases ⭐
- **log₂(n):** Try for very high-dimensional data
- **n/3:** Alternative option
- **Manual:** Only if you have specific reasons

**Example:**
```python
# Let GEE choose automatically (recommended)
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=200,
    variablesPerSplit=None  # Default: sqrt(n_features)
)

# Or specify manually
import math
n_features = 23
vars_per_split = int(math.sqrt(n_features))  # 5 for 23 features
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=200,
    variablesPerSplit=vars_per_split
)
```

---

### 3. Minimum Leaf Population (`minLeafPopulation`)

**What it does:**  
Minimum number of training samples required in a leaf node.

**Range:** 1-10 (typical: 1-5)

**Effects:**
- **Smaller values (1):**
  - ✅ Trees can fit training data better
  - ❌ Potential overfitting
  - ❌ More leaf nodes

- **Larger values (5-10):**
  - ✅ Prevents overfitting
  - ✅ Smoother decision boundaries
  - ❌ May underfit if too large

**Recommendations:**
- **Default (1):** Usually fine for EO classification ⭐
- **Noisy data:** Try 2-5
- **Small training set:** Keep at 1
- **Large training set (>1000 samples):** Can use 2-5

**Example:**
```python
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=200,
    minLeafPopulation=1  # Default
)
```

---

### 4. Bag Fraction (`bagFraction`)

**What it does:**  
Fraction of training data to use for each tree (bootstrap sampling).

**Range:** 0.1-1.0 (typical: 0.5-0.7)

**Effects:**
- **Smaller fractions (0.3-0.5):**
  - ✅ More diverse trees
  - ✅ Can reduce overfitting
  - ❌ Each tree sees less data

- **Larger fractions (0.7-1.0):**
  - ✅ Each tree better trained
  - ❌ Less tree diversity
  - ❌ Slower training

**Recommendations:**
- **Default (0.5):** Good starting point ⭐
- **Small dataset (<500 samples):** Use 0.7-0.8
- **Large dataset:** Can use 0.3-0.5
- **Imbalanced classes:** Try 0.6-0.7

**Example:**
```python
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=200,
    bagFraction=0.5
)
```

---

## Tuning Strategy

### Step-by-Step Approach

#### 1. Baseline Performance
```python
# Start with defaults
baseline = ee.Classifier.smileRandomForest(
    numberOfTrees=100,
    seed=42
).train(features=training, classProperty='class_id', inputProperties=feature_stack.bandNames())

baseline_accuracy = validation.classify(baseline).errorMatrix('class_id', 'classification').accuracy()
```

#### 2. Tune Number of Trees
```python
tree_counts = [50, 100, 200, 500]
results = {}

for n_trees in tree_counts:
    clf = ee.Classifier.smileRandomForest(numberOfTrees=n_trees, seed=42).train(
        features=training, classProperty='class_id', inputProperties=feature_stack.bandNames())
    acc = validation.classify(clf).errorMatrix('class_id', 'classification').accuracy().getInfo()
    results[n_trees] = acc

optimal_trees = max(results, key=results.get)
```

#### 3. Tune Variables Per Split (optional)
```python
import math
n_features = feature_stack.bandNames().size().getInfo()

var_options = [
    int(math.sqrt(n_features)),      # Default
    int(math.log2(n_features)),      # Conservative
    int(n_features / 3),             # Moderate
]

for vars_per_split in var_options:
    clf = ee.Classifier.smileRandomForest(
        numberOfTrees=optimal_trees,
        variablesPerSplit=vars_per_split,
        seed=42
    ).train(features=training, classProperty='class_id', inputProperties=feature_stack.bandNames())
    
    # Evaluate...
```

#### 4. Tune Bag Fraction (optional)
```python
bag_fractions = [0.3, 0.5, 0.7]

for bag_frac in bag_fractions:
    clf = ee.Classifier.smileRandomForest(
        numberOfTrees=optimal_trees,
        bagFraction=bag_frac,
        seed=42
    ).train(features=training, classProperty='class_id', inputProperties=feature_stack.bandNames())
    
    # Evaluate...
```

---

## Recommended Configurations

### Quick & Reliable (Default)
```python
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=100,
    variablesPerSplit=None,  # sqrt(n)
    minLeafPopulation=1,
    bagFraction=0.5,
    seed=42
)
```

### Production Quality (Recommended) ⭐
```python
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=200,
    variablesPerSplit=None,
    minLeafPopulation=1,
    bagFraction=0.5,
    seed=42
)
```

### High Accuracy (Slower)
```python
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=500,
    variablesPerSplit=None,
    minLeafPopulation=1,
    bagFraction=0.7,
    seed=42
)
```

### Noisy Data / Overfitting Prevention
```python
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=200,
    variablesPerSplit=None,
    minLeafPopulation=3,
    bagFraction=0.5,
    seed=42
)
```

---

## Out-of-Bag (OOB) Error

Random Forest provides a built-in validation metric using samples not used in each tree's training.

```python
# Training with OOB estimation
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=200,
    outOfBagMode=True  # Enable OOB error calculation
).train(
    features=training,
    classProperty='class_id',
    inputProperties=feature_stack.bandNames()
)

# Note: OOB error extraction in GEE requires specific handling
# Usually better to use independent validation set
```

---

## Cross-Validation

For robust accuracy estimation, use k-fold cross-validation:

```python
def kfold_cv(training_data, k=5):
    \"\"\"
    Perform k-fold cross-validation
    \"\"\"
    # Add random column for splitting
    training_data = training_data.randomColumn('random', seed=42)
    
    accuracies = []
    
    for fold in range(k):
        # Split data
        val_fold = training_data.filter(ee.Filter.gte('random', fold/k)).filter(ee.Filter.lt('random', (fold+1)/k))
        train_fold = training_data.filter(ee.Filter.lt('random', fold/k)).merge(
                      training_data.filter(ee.Filter.gte('random', (fold+1)/k)))
        
        # Train and evaluate
        clf = ee.Classifier.smileRandomForest(numberOfTrees=200).train(
            features=train_fold,
            classProperty='class_id',
            inputProperties=feature_stack.bandNames()
        )
        
        validated = val_fold.classify(clf)
        acc = validated.errorMatrix('class_id', 'classification').accuracy().getInfo()
        accuracies.append(acc)
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'fold_accuracies': accuracies
    }

# Run 5-fold CV
cv_results = kfold_cv(training, k=5)
print(f"Mean Accuracy: {cv_results['mean_accuracy']*100:.2f}% ± {cv_results['std_accuracy']*100:.2f}%")
```

---

## Tuning Tips

### Do's ✅
- Always use a fixed `seed` for reproducibility
- Tune on independent validation data (never training data)
- Start with defaults, then tune incrementally
- Monitor both accuracy and computational cost
- Use cross-validation for robust estimates
- Document your final configuration

### Don'ts ❌
- Don't over-tune on small validation sets (overfitting to validation)
- Don't use more than 500 trees (diminishing returns)
- Don't tune too many parameters simultaneously (use grid search)
- Don't ignore computational constraints
- Don't change all parameters from defaults without testing

---

## When to Stop Tuning

Stop tuning when:
1. ✅ Validation accuracy plateaus (<1% improvement)
2. ✅ Computational cost becomes prohibitive
3. ✅ Accuracy meets project requirements
4. ✅ Cross-validation shows stability

**Typical Accuracy Improvements:**
- Baseline → Optimized trees: +1-3%
- + Feature engineering: +3-7%
- + More training data: +2-5%
- + Better training data quality: +5-10%

---

## Feature Engineering vs Hyperparameter Tuning

**Impact Comparison:**

| Method | Typical Accuracy Gain | Effort | Cost |
|--------|----------------------|--------|------|
| Better training data | +5-10% | High | Low |
| Feature engineering | +3-7% | High | Medium |
| Hyperparameter tuning | +1-3% | Low | Low-Medium |
| Ensemble methods | +2-4% | Medium | High |

**Recommendation:** Focus on training data quality and features FIRST, then tune hyperparameters.

---

## Advanced: Grid Search

```python
# Define parameter grid
param_grid = {
    'numberOfTrees': [100, 200, 500],
    'bagFraction': [0.3, 0.5, 0.7],
    'minLeafPopulation': [1, 2, 5]
}

# Grid search
best_score = 0
best_params = {}

for n_trees in param_grid['numberOfTrees']:
    for bag_frac in param_grid['bagFraction']:
        for min_leaf in param_grid['minLeafPopulation']:
            
            clf = ee.Classifier.smileRandomForest(
                numberOfTrees=n_trees,
                bagFraction=bag_frac,
                minLeafPopulation=min_leaf,
                seed=42
            ).train(
                features=training,
                classProperty='class_id',
                inputProperties=feature_stack.bandNames()
            )
            
            acc = validation.classify(clf).errorMatrix('class_id', 'classification').accuracy().getInfo()
            
            if acc > best_score:
                best_score = acc
                best_params = {
                    'numberOfTrees': n_trees,
                    'bagFraction': bag_frac,
                    'minLeafPopulation': min_leaf
                }
            
            print(f"Trees={n_trees}, Bag={bag_frac}, MinLeaf={min_leaf}: {acc*100:.2f}%")

print(f"\\nBest params: {best_params}")
print(f"Best accuracy: {best_score*100:.2f}%")
```

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Belgiu, M., & Drăguț, L. (2016). Random forest in remote sensing: A review of applications. *ISPRS Journal*
- [GEE Random Forest Documentation](https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest)

---

**Last Updated:** October 2025  
**For:** CopPhil Advanced Training - Session 2

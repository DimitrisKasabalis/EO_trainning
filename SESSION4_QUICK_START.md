# Session 4 Test - Quick Start (5 minutes)

**Goal:** Get the notebook running in Colab with GPU in under 5 minutes

---

## ⚡ Speed Run Instructions

### 1️⃣ Open Colab (30 seconds)
🔗 **Go to:** [https://colab.research.google.com/](https://colab.research.google.com/)

### 2️⃣ Upload Notebook (1 minute)
**Option A - Drag & Drop:**
- Drag this file to Colab: `/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil/course_site/day2/notebooks/session4_cnn_classification_STUDENT.ipynb`

**Option B - Upload:**
- File → Upload notebook
- Navigate to the path above
- Click Open

### 3️⃣ Enable GPU (1 minute) ⚠️ CRITICAL
1. Click **Runtime** (top menu)
2. Click **Change runtime type**
3. **Hardware accelerator:** Select **GPU** ⬇️
4. Click **Save**
5. Wait for runtime to restart (~30 seconds)

### 4️⃣ Verify GPU (30 seconds)
Run the first code cell (or create a new one):
```python
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))
```

**Expected:** `[PhysicalDevice(name='/physical_device:GPU:0', ...)]`  
**If empty `[]`:** Go back to step 3, ensure GPU selected

### 5️⃣ Start Testing (2 minutes)
1. Click **Runtime** → **Run all** (or Ctrl+F9)
2. Watch for:
   - ✅ Imports succeed
   - ✅ Dataset starts downloading
   - ✅ No immediate errors

---

## 🎯 Critical Success Checkpoints

**Checkpoint 1 (5 min):** Dataset downloaded ✅  
**Checkpoint 2 (15 min):** Training started, GPU active ✅  
**Checkpoint 3 (30 min):** Training >80% accuracy ✅  
**Checkpoint 4 (45 min):** Training complete, >90% accuracy ✅  

---

## 🚨 If Something Goes Wrong

**GPU not detected:**
```
Runtime → Factory reset runtime → Change runtime type → GPU → Save
```

**Download fails:**
```
Colab internet issues - try again in 5 min or use different browser
```

**Import errors:**
```python
!pip install tensorflow==2.13.0
# Then: Runtime → Restart runtime
```

**Notebook freezes:**
```
Runtime → Interrupt execution → Restart → Run all again
```

---

## 📊 What You're Testing For

| Item | Target | How to Check |
|------|--------|--------------|
| **GPU Active** | Yes | First cell shows GPU device |
| **Download** | ~90 MB, 2-5 min | Watch progress bar |
| **Training Time** | 10-20 min | Monitor epoch progress |
| **Test Accuracy** | >90% | Final evaluation cell |
| **All Cells Run** | 45/45 | No error messages |

---

## ⏱️ Time Budget

- Setup: 5 min
- Data prep: 10 min
- Training: 15 min (with GPU)
- Evaluation: 5 min
- **Total: ~35-40 min** ✅

If it takes >60 min, GPU likely not enabled!

---

## ✅ Quick Pass Criteria

After running, check:
- [ ] GPU was detected and used
- [ ] EuroSAT downloaded (27,000 images)
- [ ] Training completed without errors
- [ ] Test accuracy >90%
- [ ] Confusion matrix displayed
- [ ] Learning curves rendered

**3+ checkmarks = PASS** ✅

---

## 📝 After Testing

1. **Document results** in test template
2. **Screenshot** the confusion matrix
3. **Note** any issues encountered
4. **Save notebook** with outputs (File → Download)

---

**Ready? Click this link to start:**  
👉 [https://colab.research.google.com/](https://colab.research.google.com/)

**Then upload:**  
`/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil/course_site/day2/notebooks/session4_cnn_classification_STUDENT.ipynb`

**Good luck! 🚀**

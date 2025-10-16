# Day 2 Reveal.js Presentations Creation Plan

**Goal:** Create 4 professional Reveal.js presentations for Day 2 based on session QMD content

**Source:** `course_site/day2/sessions/session[1-4].qmd`
**Output:** `course_site/day2/presentations/session[1-4]_presentation.qmd`

---

## Presentations to Create

### 1. Session 1: Random Forest Classification
**Source:** session1.qmd (402 lines)
**Key Topics:**
- Supervised classification workflow
- Decision trees fundamentals
- Random Forest ensemble method
- Feature importance
- Accuracy assessment
- Google Earth Engine
- Palawan case study (NRM)

**Slides:** ~40-50 slides

---

### 2. Session 2: Advanced Palawan Lab
**Source:** session2.qmd (513 lines)
**Key Topics:**
- Advanced feature engineering (GLCM texture)
- Multi-temporal composites
- Hyperparameter tuning
- Change detection (2020-2024)
- Protected area monitoring
- 8-class detailed classification

**Slides:** ~35-45 slides

---

### 3. Session 3: Deep Learning & CNNs
**Source:** session3.qmd (1,328 lines - very detailed)
**Key Topics:**
- ML to DL transition
- Neural network fundamentals
- Perceptron, activation functions
- Convolutional operations
- CNN architectures (LeNet, VGG, ResNet, U-Net)
- Transfer learning
- Philippine EO applications

**Slides:** ~50-60 slides

---

### 4. Session 4: CNN Hands-on Lab
**Source:** session4.qmd (942 lines)
**Key Topics:**
- EuroSAT dataset preparation
- Building CNNs from scratch
- Training with TensorFlow/Keras
- Transfer learning with ResNet50
- Model evaluation
- Philippine applications

**Slides:** ~40-50 slides

---

## Presentation Style Guide

**Match Day 1 presentations:**
- Use `revealjs` format
- Include `theme: [default, custom.scss]`
- Add slide numbers, footer with session info
- Use incremental reveals for bullet points
- Include mermaid diagrams for workflows
- Add callout boxes for key concepts
- Use columns for side-by-side content
- Include Philippine context throughout

**Structure:**
```yaml
---
title: "Session X: Title"
subtitle: "Subtitle"
author: "CopPhil Advanced Training Program - DAY 2"
format:
  revealjs:
    theme: [default, custom.scss]
    slide-number: true
    chalkboard: true
    preview-links: auto
    footer: "DAY 2 - Session X | Topic"
    transition: fade
    background-transition: fade
    width: 1920
    height: 1080
    margin: 0.1
---
```

---

## Creating Presentations Now

I'll create all 4 presentations with proper content extracted from the session QMDs...

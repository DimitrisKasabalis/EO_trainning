#!/usr/bin/env python3
"""
Fix Session 4 CNN Notebook - Augmentation Visualization Bug
Corrects the numpy scalar indexing issue in the visualization cell
"""

import json
import sys
from pathlib import Path

def fix_augmentation_cell(notebook_path):
    """
    Fix the augmentation visualization cell in the Session 4 notebook
    
    Parameters:
    -----------
    notebook_path : str or Path
        Path to the notebook file
    """
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the problematic cell
    fixed = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Check if this is the augmentation visualization cell
            source = ''.join(cell['source'])
            if 'Show original vs augmented' in source and 'data_augmentation' in source:
                print(f"Found augmentation visualization cell...")
                
                # Fixed code
                fixed_source = [
                    "# Show original vs augmented\n",
                    "sample_image, sample_label = next(iter(ds_train))\n",
                    "\n",
                    "# Convert label to integer for indexing\n",
                    "label_idx = int(sample_label.numpy())\n",
                    "\n",
                    "fig, axes = plt.subplots(2, 4, figsize=(14, 7))\n",
                    "\n",
                    "# Original\n",
                    "axes[0, 0].imshow(sample_image.numpy())\n",
                    "axes[0, 0].set_title('Original', fontweight='bold')\n",
                    "axes[0, 0].axis('off')\n",
                    "\n",
                    "# Augmented versions\n",
                    "for idx in range(1, 8):\n",
                    "    row = idx // 4\n",
                    "    col = idx % 4\n",
                    "    \n",
                    "    augmented = data_augmentation(tf.expand_dims(sample_image, 0), training=True)[0]\n",
                    "    axes[row, col].imshow(augmented.numpy())\n",
                    "    axes[row, col].set_title(f'Augmented {idx}', fontweight='bold')\n",
                    "    axes[row, col].axis('off')\n",
                    "\n",
                    "# Fixed: Use label_idx instead of sample_label.numpy()\n",
                    "plt.suptitle(f'Data Augmentation Examples\\nClass: {class_names[label_idx]}',\n",
                    "             fontsize=14, fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"\\n✓ Augmentation creates realistic variations\")"
                ]
                
                cell['source'] = fixed_source
                fixed = True
                print("✓ Applied fix to augmentation visualization cell")
                break
    
    if not fixed:
        print("⚠️  Could not find the augmentation visualization cell")
        return False
    
    # Save fixed notebook
    output_path = notebook_path.parent / f"{notebook_path.stem}_FIXED{notebook_path.suffix}"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"\n✅ Fixed notebook saved to: {output_path}")
    return True

if __name__ == "__main__":
    # Paths to the notebook
    notebook_paths = [
        Path("/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil/DAY_2/session4/notebooks/session4_cnn_classification_STUDENT.ipynb"),
        Path("/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil/course_site/day2/notebooks/session4_cnn_classification_STUDENT.ipynb")
    ]
    
    print("=" * 60)
    print("Session 4 Notebook Fix Script")
    print("=" * 60)
    print("\nFixing augmentation visualization bug...")
    print("Issue: numpy scalar indexing in class_names lookup")
    print("Fix: Explicit int() conversion\n")
    
    for notebook_path in notebook_paths:
        if notebook_path.exists():
            print(f"\nProcessing: {notebook_path}")
            success = fix_augmentation_cell(notebook_path)
            if success:
                print(f"✅ Successfully fixed notebook")
            else:
                print(f"❌ Failed to fix notebook")
        else:
            print(f"\n⚠️  Notebook not found: {notebook_path}")
    
    print("\n" + "=" * 60)
    print("Fix complete!")
    print("=" * 60)

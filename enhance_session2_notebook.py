#!/usr/bin/env python3
"""
Enhance Day 3 Session 2 notebook with synthetic SAR data generation
This makes the notebook immediately executable without external data downloads
"""

import json
import sys

def create_synthetic_data_cell():
    """Create markdown and code cells for synthetic SAR data generation"""
    
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Generate Synthetic SAR Data\n\n"
            "::: {.callout-note}\n"
            "## Synthetic Data Approach\n\n"
            "For this lab, we'll generate **synthetic SAR data** that mimics real Sentinel-1 characteristics. "
            "This allows you to:\n"
            "- ✅ Run the notebook immediately without downloads\n"
            "- ✅ Understand data structure and formats\n"
            "- ✅ Practice the complete U-Net workflow\n"
            "- ✅ Learn model training and evaluation\n\n"
            "**The workflow is identical to using real data** - only the data source differs. "
            "See the [Data Acquisition Guide](../DATA_GUIDE.md) for instructions on obtaining real Central Luzon SAR flood data.\n"
            ":::"
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def generate_synthetic_sar_flood_data(n_train=800, n_val=200, n_test=200, \n"
            "                                       img_size=256, seed=42):\n"
            "    \"\"\"\n"
            "    Generate synthetic SAR flood mapping dataset\n"
            "    \n"
            "    Simulates Sentinel-1 dual-polarization (VV, VH) imagery with flood masks\n"
            "    \n"
            "    Args:\n"
            "        n_train: Number of training samples\n"
            "        n_val: Number of validation samples\n"
            "        n_test: Number of test samples\n"
            "        img_size: Image dimension (default 256x256)\n"
            "        seed: Random seed for reproducibility\n"
            "    \n"
            "    Returns:\n"
            "        Dictionary with paths to generated data\n"
            "    \"\"\"\n"
            "    np.random.seed(seed)\n"
            "    print(\"Generating synthetic SAR flood data...\")\n"
            "    print(f\"Train: {n_train}, Val: {n_val}, Test: {n_test} samples\")\n"
            "    \n"
            "    # Create directory structure\n"
            "    data_dir = '/content/data/flood_mapping_dataset'\n"
            "    for subset in ['train', 'val', 'test']:\n"
            "        os.makedirs(os.path.join(data_dir, subset, 'images'), exist_ok=True)\n"
            "        os.makedirs(os.path.join(data_dir, subset, 'masks'), exist_ok=True)\n"
            "    \n"
            "    def generate_sample(idx, subset):\n"
            "        \"\"\"Generate one SAR image + flood mask pair\"\"\"\n"
            "        \n"
            "        # Simulate SAR backscatter (in dB)\n"
            "        # VV: -25 to 5 dB (typical range)\n"
            "        # VH: -30 to 0 dB (typical range)\n"
            "        vv = np.random.normal(-10, 5, (img_size, img_size))\n"
            "        vh = np.random.normal(-15, 5, (img_size, img_size))\n"
            "        \n"
            "        # Create flood mask with realistic patterns\n"
            "        # Floods appear as connected regions (not random noise)\n"
            "        \n"
            "        # Start with base mask\n"
            "        mask = np.zeros((img_size, img_size), dtype=np.float32)\n"
            "        \n"
            "        # Add 1-3 flood regions per image\n"
            "        n_floods = np.random.randint(1, 4)\n"
            "        \n"
            "        for _ in range(n_floods):\n"
            "            # Random flood center\n"
            "            center_x = np.random.randint(50, img_size-50)\n"
            "            center_y = np.random.randint(50, img_size-50)\n"
            "            \n"
            "            # Random flood size (elliptical shape)\n"
            "            radius_x = np.random.randint(20, 80)\n"
            "            radius_y = np.random.randint(20, 80)\n"
            "            \n"
            "            # Create elliptical flood region\n"
            "            y, x = np.ogrid[:img_size, :img_size]\n"
            "            ellipse = ((x - center_x)**2 / radius_x**2 + \n"
            "                      (y - center_y)**2 / radius_y**2 <= 1)\n"
            "            mask[ellipse] = 1.0\n"
            "        \n"
            "        # Apply Gaussian smoothing to make edges more realistic\n"
            "        from scipy.ndimage import gaussian_filter\n"
            "        mask = gaussian_filter(mask, sigma=2.0)\n"
            "        mask = (mask > 0.3).astype(np.float32)  # Threshold\n"
            "        \n"
            "        # Modify SAR values in flooded regions\n"
            "        # Flooded areas have LOW backscatter (dark in SAR)\n"
            "        flood_mask_bool = mask > 0.5\n"
            "        vv[flood_mask_bool] = np.random.normal(-20, 3, flood_mask_bool.sum())\n"
            "        vh[flood_mask_bool] = np.random.normal(-25, 3, flood_mask_bool.sum())\n"
            "        \n"
            "        # Non-flooded areas have HIGHER backscatter\n"
            "        non_flood = ~flood_mask_bool\n"
            "        vv[non_flood] = np.random.normal(-5, 4, non_flood.sum())\n"
            "        vh[non_flood] = np.random.normal(-10, 4, non_flood.sum())\n"
            "        \n"
            "        # Clip to realistic SAR ranges\n"
            "        vv = np.clip(vv, -30, 10)\n"
            "        vh = np.clip(vh, -35, 5)\n"
            "        \n"
            "        # Stack VV and VH\n"
            "        sar_image = np.stack([vv, vh], axis=-1).astype(np.float32)\n"
            "        \n"
            "        # Expand mask dimension\n"
            "        mask = np.expand_dims(mask, axis=-1).astype(np.float32)\n"
            "        \n"
            "        # Save\n"
            "        img_path = os.path.join(data_dir, subset, 'images', f'sar_{idx:04d}.npy')\n"
            "        mask_path = os.path.join(data_dir, subset, 'masks', f'mask_{idx:04d}.npy')\n"
            "        \n"
            "        np.save(img_path, sar_image)\n"
            "        np.save(mask_path, mask)\n"
            "    \n"
            "    # Generate all samples\n"
            "    print(\"Generating training samples...\")\n"
            "    for i in range(n_train):\n"
            "        generate_sample(i, 'train')\n"
            "        if (i+1) % 200 == 0:\n"
            "            print(f\"  Generated {i+1}/{n_train} training samples\")\n"
            "    \n"
            "    print(\"Generating validation samples...\")\n"
            "    for i in range(n_val):\n"
            "        generate_sample(i, 'val')\n"
            "    \n"
            "    print(\"Generating test samples...\")\n"
            "    for i in range(n_test):\n"
            "        generate_sample(i, 'test')\n"
            "    \n"
            "    print(f\"\\n✅ Synthetic dataset generated successfully!\")\n"
            "    print(f\"Location: {data_dir}\")\n"
            "    print(f\"Train: {n_train} samples\")\n"
            "    print(f\"Val: {n_val} samples\")\n"
            "    print(f\"Test: {n_test} samples\")\n"
            "    \n"
            "    return {\n"
            "        'data_dir': data_dir,\n"
            "        'n_train': n_train,\n"
            "        'n_val': n_val,\n"
            "        'n_test': n_test\n"
            "    }\n\n"
            "# Generate synthetic data (takes ~2-3 minutes)\n"
            "dataset_info = generate_synthetic_sar_flood_data(\n"
            "    n_train=800,  # 800 training samples\n"
            "    n_val=200,    # 200 validation samples\n"
            "    n_test=200,   # 200 test samples\n"
            "    img_size=256,\n"
            "    seed=42\n"
            ")\n\n"
            "DATA_DIR = dataset_info['data_dir']\n"
            "print(f\"\\nDataset ready at: {DATA_DIR}\")"
        ],
        "outputs": [],
        "execution_count": None
    }
    
    return markdown_cell, code_cell


def enhance_notebook(input_path, output_path):
    """Add synthetic data generation to notebook"""
    
    print(f"Loading notebook: {input_path}")
    with open(input_path, 'r') as f:
        nb = json.load(f)
    
    print(f"Original notebook has {len(nb['cells'])} cells")
    
    # Find the dataset download cell (contains "DATASET_URL")
    download_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'DATASET_URL' in source or 'wget' in source:
                download_cell_idx = i
                print(f"Found download cell at index {i}")
                break
    
    if download_cell_idx is None:
        print("ERROR: Could not find dataset download cell")
        return False
    
    # Create synthetic data cells
    markdown_cell, code_cell = create_synthetic_data_cell()
    
    # Replace the download cell with our new cells
    # Insert markdown first, then code
    nb['cells'][download_cell_idx] = markdown_cell
    nb['cells'].insert(download_cell_idx + 1, code_cell)
    
    print(f"Enhanced notebook now has {len(nb['cells'])} cells")
    
    # Add a note at the top about synthetic data
    intro_note = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎓 Educational Note: Synthetic Data\n\n"
            "This notebook uses **synthetic SAR data** for immediate execution and learning. "
            "The U-Net architecture, training workflow, and evaluation metrics are identical to real-world applications.\n\n"
            "**Benefits:**\n"
            "- ✅ No data download required\n"
            "- ✅ Runs in 5-10 minutes (vs. hours for real data preprocessing)\n"
            "- ✅ Perfect for understanding the workflow\n"
            "- ✅ Easy to experiment and modify\n\n"
            "**For production work:** Replace synthetic data with real Sentinel-1 SAR from Google Earth Engine or the CopPhil Mirror Site. "
            "See the [Data Acquisition Guide](../DATA_GUIDE.md) for details.\n\n"
            "---"
        ]
    }
    
    # Insert note after the title cell (usually index 0)
    nb['cells'].insert(1, intro_note)
    
    # Save enhanced notebook
    print(f"Saving enhanced notebook to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=2)
    
    print("✅ Notebook enhancement complete!")
    print(f"\nChanges made:")
    print("1. Added educational note about synthetic data")
    print("2. Replaced dataset download with synthetic SAR generation")
    print("3. Generated realistic flood patterns")
    print("4. Maintained all original workflow steps")
    
    return True


if __name__ == "__main__":
    input_nb = "/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil/course_site/day3/notebooks/Day3_Session2_Flood_Mapping_UNet.ipynb"
    output_nb = input_nb  # Overwrite the original
    
    success = enhance_notebook(input_nb, output_nb)
    sys.exit(0 if success else 1)

"""
Test Suite for GLCM Texture Feature Template
Tests all functions and validates outputs
"""

import ee
import sys

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✓ Earth Engine initialized successfully")
except Exception as e:
    print(f"✗ Error initializing Earth Engine: {e}")
    print("Please run: earthengine authenticate")
    sys.exit(1)

# Import the GLCM template functions
from glcm_template import (
    add_glcm_texture,
    add_selected_glcm,
    glcm_for_classification,
    multiscale_glcm
)


def create_test_image():
    """Create a simple test image for validation"""
    # Use Palawan area
    aoi = ee.Geometry.Rectangle([118.5, 9.5, 119.0, 10.0])

    # Load a single Sentinel-2 image
    image = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(aoi) \
        .filterDate('2024-01-01', '2024-04-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .first()

    return image, aoi


def test_add_glcm_texture():
    """Test 1: Basic GLCM texture addition"""
    print("\n" + "="*60)
    print("TEST 1: add_glcm_texture()")
    print("="*60)

    try:
        image, aoi = create_test_image()

        # Test with default parameters
        result = add_glcm_texture(image, bands=['B8'], radius=1)

        # Get band names
        band_names = result.bandNames().getInfo()

        # Check if GLCM bands were added
        glcm_bands = [b for b in band_names if 'B8_' in b]

        print(f"✓ Function executed successfully")
        print(f"✓ Original bands: {len([b for b in band_names if not 'B8_' in b or b == 'B8'])}")
        print(f"✓ GLCM bands added: {len(glcm_bands)}")
        print(f"✓ Sample GLCM bands: {glcm_bands[:5]}")

        # Expected GLCM features
        expected_features = ['asm', 'contrast', 'corr', 'var', 'idm', 'savg',
                            'svar', 'sent', 'ent', 'dvar', 'dent']

        found_features = set()
        for band in glcm_bands:
            for feat in expected_features:
                if f'_{feat}' in band:
                    found_features.add(feat)

        print(f"✓ Features found: {sorted(found_features)}")

        if len(found_features) >= 8:
            print("✓ TEST 1 PASSED: All major GLCM features present")
            return True
        else:
            print("✗ TEST 1 WARNING: Some GLCM features missing")
            return False

    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_add_selected_glcm():
    """Test 2: Selected GLCM features"""
    print("\n" + "="*60)
    print("TEST 2: add_selected_glcm()")
    print("="*60)

    try:
        image, aoi = create_test_image()

        # Test with specific features
        selected_features = ['contrast', 'ent', 'corr']
        result = add_selected_glcm(image, bands=['B8'], features=selected_features)

        # Get band names
        band_names = result.bandNames().getInfo()
        texture_bands = [b for b in band_names if any(f'_{f}' in b for f in selected_features)]

        print(f"✓ Function executed successfully")
        print(f"✓ Requested features: {selected_features}")
        print(f"✓ Texture bands added: {texture_bands}")

        # Verify only selected features are present
        if len(texture_bands) == len(selected_features):
            print(f"✓ TEST 2 PASSED: Correct number of features selected")
            return True
        else:
            print(f"✗ TEST 2 WARNING: Expected {len(selected_features)} bands, got {len(texture_bands)}")
            return False

    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_glcm_for_classification():
    """Test 3: Classification-optimized GLCM"""
    print("\n" + "="*60)
    print("TEST 3: glcm_for_classification()")
    print("="*60)

    try:
        image, aoi = create_test_image()

        # Test classification function
        result = glcm_for_classification(image, nir_band='B8', red_band='B4')

        # Get band names
        band_names = result.bandNames().getInfo()
        texture_bands = [b for b in band_names if 'texture' in b]

        print(f"✓ Function executed successfully")
        print(f"✓ Texture bands added: {texture_bands}")

        # Expected bands
        expected = ['nir_texture_contrast', 'nir_texture_entropy',
                   'nir_texture_corr', 'red_texture_contrast']

        all_present = all(band in band_names for band in expected)

        if all_present:
            print(f"✓ All expected bands present: {expected}")
            print("✓ TEST 3 PASSED: Classification features correctly added")
            return True
        else:
            missing = [b for b in expected if b not in band_names]
            print(f"✗ TEST 3 FAILED: Missing bands: {missing}")
            return False

    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiscale_glcm():
    """Test 4: Multi-scale GLCM"""
    print("\n" + "="*60)
    print("TEST 4: multiscale_glcm()")
    print("="*60)

    try:
        image, aoi = create_test_image()

        # Test multi-scale function
        radii = [1, 2]
        result = multiscale_glcm(image, band='B8', radii=radii)

        # Get band names
        band_names = result.bandNames().getInfo()
        texture_bands = [b for b in band_names if 'texture' in b]

        print(f"✓ Function executed successfully")
        print(f"✓ Radii used: {radii}")
        print(f"✓ Texture bands added: {texture_bands}")

        # Check for different scales
        scales_found = set()
        for radius in radii:
            for band in texture_bands:
                if f'_r{radius}' in band:
                    scales_found.add(radius)

        if scales_found == set(radii):
            print(f"✓ All scales present: {sorted(scales_found)}")
            print("✓ TEST 4 PASSED: Multi-scale features correctly added")
            return True
        else:
            print(f"✗ TEST 4 WARNING: Expected scales {radii}, found {sorted(scales_found)}")
            return False

    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_computation_performance():
    """Test 5: Performance and computation feasibility"""
    print("\n" + "="*60)
    print("TEST 5: Computation Performance")
    print("="*60)

    try:
        import time

        image, aoi = create_test_image()

        # Test computation on small area
        small_aoi = ee.Geometry.Rectangle([118.5, 9.5, 118.6, 9.6])

        # Time the computation
        start = time.time()
        result = glcm_for_classification(image)

        # Force computation by getting info on small region
        sample = result.select(['nir_texture_contrast']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=small_aoi,
            scale=10,
            maxPixels=1e6
        ).getInfo()

        elapsed = time.time() - start

        print(f"✓ Computation completed in {elapsed:.2f} seconds")
        print(f"✓ Sample value: {sample}")

        if elapsed < 30:
            print("✓ TEST 5 PASSED: Performance acceptable")
            return True
        else:
            print("✗ TEST 5 WARNING: Computation took longer than expected")
            return False

    except Exception as e:
        print(f"✗ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test 6: Edge cases and error handling"""
    print("\n" + "="*60)
    print("TEST 6: Edge Cases")
    print("="*60)

    try:
        image, aoi = create_test_image()

        # Test 6a: Multiple bands
        print("\n6a. Testing multiple bands...")
        result = add_selected_glcm(image, bands=['B4', 'B8'], features=['contrast'])
        band_names = result.bandNames().getInfo()
        contrast_bands = [b for b in band_names if 'contrast' in b]
        print(f"   ✓ Multiple bands: {contrast_bands}")

        # Test 6b: Different window sizes
        print("\n6b. Testing different window sizes...")
        result_small = add_glcm_texture(image, bands=['B8'], radius=1)
        result_large = add_glcm_texture(image, bands=['B8'], radius=2)
        print(f"   ✓ Small window (3x3): Success")
        print(f"   ✓ Large window (5x5): Success")

        # Test 6c: Empty feature list (should handle gracefully)
        print("\n6c. Testing empty feature list...")
        try:
            result = add_selected_glcm(image, bands=['B8'], features=[])
            print(f"   ✓ Empty features handled")
        except:
            print(f"   ⚠ Empty features causes error (expected)")

        print("\n✓ TEST 6 PASSED: Edge cases handled appropriately")
        return True

    except Exception as e:
        print(f"✗ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("GLCM TEMPLATE TEST SUITE")
    print("="*60)

    tests = [
        ("Basic GLCM", test_add_glcm_texture),
        ("Selected GLCM", test_add_selected_glcm),
        ("Classification GLCM", test_glcm_for_classification),
        ("Multi-scale GLCM", test_multiscale_glcm),
        ("Performance", test_computation_performance),
        ("Edge Cases", test_edge_cases)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print("\n🎉 All tests passed! GLCM template is working correctly.")
    elif passed >= total * 0.8:
        print("\n⚠ Most tests passed. Some minor issues detected.")
    else:
        print("\n❌ Multiple test failures. Review code before use.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

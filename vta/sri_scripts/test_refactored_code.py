"""
Test script for refactored candidate set execution.
This script validates that the refactored code works correctly.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import numpy as np
import torch

from execute_candidate_set_refactored import (
    Config,
    setup_external_imports,
    load_arch_mapping,
    load_candidate_set_from_experiments,
    load_ofa_model,
    pytorch_to_relay,
    PACK_DICT
)


def test_config():
    """Test configuration dataclass."""
    print("Testing Config...")
    config = Config()

    assert config.device in ["vta", "arm_cpu"]
    assert config.model_name in PACK_DICT
    assert config.opt_level == 3
    assert config.skip_conv_layers == [0]

    print("✓ Config test passed")


def test_external_imports():
    """Test external module imports."""
    print("\nTesting external imports...")

    try:
        OFADynamicResnetAllMod, StaticResNetFromArch = setup_external_imports(
            "/home/srchand/Desktop/research/OFA_Obfs"
        )
        assert OFADynamicResnetAllMod is not None
        assert StaticResNetFromArch is not None
        print("✓ External imports test passed")
    except Exception as e:
        print(f"✗ External imports test failed: {e}")
        print("  (This is expected if OFA_Obfs is not available)")


def test_arch_loading():
    """Test architecture loading functions."""
    print("\nTesting architecture loading...")

    config = Config()

    try:
        # Test load_arch_mapping
        arch_mapping = load_arch_mapping(config.arch_config_json)
        assert isinstance(arch_mapping, dict)
        assert len(arch_mapping) > 0

        # Check structure of first architecture
        first_arch = next(iter(arch_mapping.values()))
        assert 'residual_depth_list' in first_arch
        assert 'out_channel_setting_list' in first_arch

        print(f"  Loaded {len(arch_mapping)} architectures")
        print("✓ Architecture loading test passed")

    except FileNotFoundError:
        print("✗ Architecture loading test skipped (files not found)")
    except Exception as e:
        print(f"✗ Architecture loading test failed: {e}")


def test_candidate_set_loading():
    """Test candidate set loading."""
    print("\nTesting candidate set loading...")

    config = Config()

    try:
        arch_mapping, model_ids = load_candidate_set_from_experiments(
            config.candidate_set_json,
            config.arch_config_json,
            config.experiment_name
        )

        assert isinstance(arch_mapping, dict)
        assert isinstance(model_ids, list)
        assert len(model_ids) > 0
        assert all(model_id in arch_mapping for model_id in model_ids)

        print(f"  Loaded {len(model_ids)} models from experiment '{config.experiment_name}'")
        print("✓ Candidate set loading test passed")

    except FileNotFoundError:
        print("✗ Candidate set loading test skipped (files not found)")
    except Exception as e:
        print(f"✗ Candidate set loading test failed: {e}")


def test_ofa_model_loading():
    """Test OFA model loading."""
    print("\nTesting OFA model loading...")

    config = Config()

    try:
        setup_external_imports(config.external_repo_root)
        ofa_net = load_ofa_model(config.model_path)

        assert ofa_net is not None

        # Test setting active subnet
        test_arch = {
            'residual_depth_list': [[0, 0], [2, 2, 2], [2, 2, 2, 2], [2, 2, 2]],
            'out_channel_setting_list': [
                [1, 1],
                [1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1]
            ],
            'decomp_type_list': [
                [[1], [1]],
                [[1], [1], [1]],
                [[1], [1], [1], [1]],
                [[1], [1], [1]]
            ]
        }

        ofa_net.set_active_subnet(test_arch)

        # Test forward pass
        ofa_net.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            output = ofa_net(test_input)
            assert output.shape[0] == 1
            assert output.shape[1] == 10  # num_classes

        print("✓ OFA model loading test passed")

    except FileNotFoundError:
        print("✗ OFA model loading test skipped (model file not found)")
    except ImportError:
        print("✗ OFA model loading test skipped (OFA modules not available)")
    except Exception as e:
        print(f"✗ OFA model loading test failed: {e}")


def test_pytorch_to_relay():
    """Test PyTorch to Relay conversion."""
    print("\nTesting PyTorch to Relay conversion...")

    try:
        import torch.nn as nn
        import tvm
        from tvm import relay

        # Create simple PyTorch model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = SimpleModel()
        input_shape = [1, 3, 32, 32]

        # Convert to Relay
        mod, params = pytorch_to_relay(model, input_shape, "input0")

        assert mod is not None
        assert isinstance(params, dict)
        assert len(params) > 0

        print(f"  Converted model with {len(params)} parameters")
        print("✓ PyTorch to Relay conversion test passed")

    except ImportError:
        print("✗ PyTorch to Relay test skipped (TVM not available)")
    except Exception as e:
        print(f"✗ PyTorch to Relay conversion test failed: {e}")


def test_pack_dict():
    """Test pack dictionary completeness."""
    print("\nTesting pack dictionary...")

    assert "resnet18" in PACK_DICT
    assert len(PACK_DICT["resnet18"]) == 2
    assert PACK_DICT["resnet18"][0] == "nn.max_pool2d"
    assert PACK_DICT["resnet18"][1] == "nn.adaptive_avg_pool2d"

    print(f"  Pack dict contains {len(PACK_DICT)} model types")
    print("✓ Pack dictionary test passed")


def test_imagenette_loader():
    """Test ImageNette data loader."""
    print("\nTesting ImageNette data loader...")

    from execute_candidate_set_refactored import ImageNetteDataLoader

    config = Config()

    try:
        loader = ImageNetteDataLoader(config.imagenette_base_dir, batch_size=1)

        assert len(loader.classes) == 10
        assert len(loader.image_paths) == 10

        # Test loading first image (without display)
        image_data = loader.load_and_preprocess(0, show_image=False)

        assert image_data.shape == (1, 3, 224, 224)
        assert isinstance(image_data, np.ndarray)

        print("✓ ImageNette loader test passed")

    except FileNotFoundError:
        print("✗ ImageNette loader test skipped (images not found)")
    except Exception as e:
        print(f"✗ ImageNette loader test failed: {e}")


def test_compiled_model_structure():
    """Test CompiledModel dataclass."""
    print("\nTesting CompiledModel structure...")

    from execute_candidate_set_refactored import CompiledModel
    import tvm

    # Create mock compiled model
    model = CompiledModel(
        model_id="test_model",
        graph="{}",
        lib=None,
        params={"param1": tvm.nd.array(np.zeros((3, 3)))}
    )

    assert model.model_id == "test_model"
    assert model.graph == "{}"
    assert "param1" in model.params
    assert model.remote_lib is None

    print("✓ CompiledModel structure test passed")


def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("Running Refactored Code Tests")
    print("="*80)

    start_time = time.time()

    # Run tests
    test_config()
    test_pack_dict()
    test_external_imports()
    test_arch_loading()
    test_candidate_set_loading()
    test_ofa_model_loading()
    test_pytorch_to_relay()
    test_imagenette_loader()
    test_compiled_model_structure()

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print(f"All tests completed in {elapsed:.2f}s")
    print("="*80)
    print("\nNote: Some tests may be skipped if required files are not available.")
    print("This is expected in a minimal testing environment.")


if __name__ == "__main__":
    run_all_tests()


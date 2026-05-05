"""Diagnostic script: probe the osimulator Gazelle optical model interface.

Usage:
    python test/inspect_optical_simulator.py
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def load_model() -> Any:
    """Load the optical model and return it."""
    from osimulator.api import load_gazelle_model

    return load_gazelle_model()


def inspect_model_object(model: Any) -> None:
    """Print general metadata about the loaded model."""
    print("=" * 60)
    print("1) Model object info")
    print("=" * 60)
    print(f"  type(model):        {type(model)}")
    print(f"  model:              {model}")
    print(f"  callable?           {callable(model)}")

    attrs = sorted([a for a in dir(model) if not a.startswith("_")])
    print(f"  public attributes:  {attrs}")

    # Try known Gazelle interface
    for attr in [
        "n_phases",
        "n_wavelengths",
        "input_size",
        "output_size",
        "weight_size",
    ]:
        if hasattr(model, attr):
            print(f"  model.{attr}: {getattr(model, attr)!r}")


def test_basic_call(model: Any) -> None:
    """Test the most basic call with tiny known values."""
    print()
    print("=" * 60)
    print("2) Basic matmul: [1, 1, 4] x [1, 4, 2]")
    print("=" * 60)

    # Activation: shape [B, M, K] = [1, 1, 4]
    # Weight:    shape [B, K, N] = [1, 4, 2]   (broadcast from [K, N])
    a = np.array([[[1, 2, 3, 4]]], dtype=np.int32)
    w = np.array([[[1, 0], [0, 1], [2, 0], [0, 2]]], dtype=np.int32)

    print(f"  input a shape:  {a.shape}, dtype: {a.dtype}")
    print(f"  input a values: {a}")
    print(f"  input w shape:  {w.shape}, dtype: {w.dtype}")
    print(f"  input w values: {w}")

    try:
        result = model(a, w, inputType="uint4")
        print(f"  output type:    {type(result)}")
        if hasattr(result, "numpy"):
            arr = result.numpy()
        else:
            arr = np.asarray(result)
        print(f"  output shape:   {arr.shape}, dtype: {arr.dtype}")
        print(f"  output values:  {arr}")
        # Expected: [[1*1+2*0+3*2+4*0, 1*0+2*1+3*0+4*2]] = [[7, 10]]
        print(f"  expected:       [[7, 10]]")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


def test_input_types(model: Any) -> None:
    """Enumerate supported inputType values."""
    print()
    print("=" * 60)
    print("3) Supported inputType values")
    print("=" * 60)

    a = np.ones((1, 1, 4), dtype=np.int32)
    w = np.ones((1, 4, 2), dtype=np.int32)

    for itype in ["uint4", "uint8", "int4", "int8"]:
        try:
            result = model(a, w, inputType=itype)
            arr = result.numpy() if hasattr(result, "numpy") else np.asarray(result)
            print(f"  inputType={itype:6s}  OK  output_shape={arr.shape}  output_range=[{arr.min():.1f}, {arr.max():.1f}]")
        except Exception as e:
            print(f"  inputType={itype:6s}  FAIL  {type(e).__name__}: {str(e)[:80]}")


def test_shapes(model: Any) -> None:
    """Test various input shapes to probe dimension constraints."""
    print()
    print("=" * 60)
    print("4) Shape compatibility")
    print("=" * 60)

    test_cases = [
        # (a_shape, w_shape, description)
        ((1, 1, 4),   (1, 4, 4),   "B=1, M=1, K=4, N=4"),
        ((2, 1, 4),   (1, 4, 4),   "B=2, M=1, K=4, N=4 (batched)"),
        ((1, 1, 784), (1, 784, 80), "MNIST fc1: [1,1,784] x [1,784,80]"),
        ((1, 1, 80),  (1, 80, 20),  "MNIST fc2: [1,1,80]  x [1,80,20]"),
        ((1, 1, 20),  (1, 20, 10),  "MNIST fc3: [1,1,20]  x [1,20,10]"),
        ((128, 1, 784), (1, 784, 80), "batch=128 fc1"),
        ((1, 1, 16),  (1, 16, 16),  "16x16 square matmul"),
        ((1, 1, 64),  (1, 64, 64),  "64x64 matmul"),
    ]

    for a_shape, w_shape, desc in test_cases:
        a = np.ones(a_shape, dtype=np.int32)
        w = np.ones(w_shape, dtype=np.int32)
        try:
            t0 = time.perf_counter()
            result = model(a, w, inputType="uint8")
            elapsed = time.perf_counter() - t0
            arr = result.numpy() if hasattr(result, "numpy") else np.asarray(result)
            print(f"  {desc:35s} OK  out={arr.shape}  time={elapsed*1e6:.0f} us")
        except Exception as e:
            print(f"  {desc:35s} FAIL  {type(e).__name__}: {str(e)[:80]}")


def test_value_range(model: Any) -> None:
    """Explore what value ranges the simulator accepts."""
    print()
    print("=" * 60)
    print("5) Input value range tolerance")
    print("=" * 60)

    a = np.zeros((1, 1, 4), dtype=np.int32)
    w = np.ones((1, 4, 2), dtype=np.int32)

    print("  --- uint4 (expected 0–15) ---")
    for val in [-1, 0, 1, 7, 8, 15, 16, 127, 255]:
        a[0, 0, :] = val
        try:
            result = model(a, w, inputType="uint4")
            arr = result.numpy() if hasattr(result, "numpy") else np.asarray(result)
            print(f"  input={val:4d}  OK  output_sum={arr.sum():.1f}")
        except Exception as e:
            print(f"  input={val:4d}  FAIL  {type(e).__name__}: {str(e)[:60]}")

    print("  --- uint8 (expected 0–255) ---")
    for val in [-1, 0, 1, 127, 128, 255, 256]:
        a[0, 0, :] = val
        try:
            result = model(a, w, inputType="uint8")
            arr = result.numpy() if hasattr(result, "numpy") else np.asarray(result)
            print(f"  input={val:4d}  OK  output_sum={arr.sum():.1f}")
        except Exception as e:
            print(f"  input={val:4d}  FAIL  {type(e).__name__}: {str(e)[:60]}")


def test_dtype_tolerance(model: Any) -> None:
    """Check what numpy dtypes are accepted."""
    print()
    print("=" * 60)
    print("6) Input dtype tolerance")
    print("=" * 60)

    a_int32 = np.ones((1, 1, 4), dtype=np.int32)
    w_int32 = np.ones((1, 4, 2), dtype=np.int32)

    for dtype_name, dtype in [
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
        ("int64", np.int64),
        ("uint8", np.uint8),
        ("uint32", np.uint32),
        ("float32", np.float32),
        ("float64", np.float64),
    ]:
        try:
            a = a_int32.astype(dtype)
            w = w_int32.astype(dtype)
            result = model(a, w, inputType="uint4")
            arr = result.numpy() if hasattr(result, "numpy") else np.asarray(result)
            print(f"  dtype={dtype.__name__:8s}  OK  out_dtype={arr.dtype}")
        except Exception as e:
            print(f"  dtype={dtype.__name__:8s}  FAIL  {type(e).__name__}: {str(e)[:60]}")


def test_output_properties(model: Any) -> None:
    """Inspect the output object."""
    print()
    print("=" * 60)
    print("7) Output object details")
    print("=" * 60)

    a = np.arange(8, dtype=np.int32).reshape(1, 1, 8)
    w = np.eye(8, dtype=np.int32).reshape(1, 8, 8)

    result = model(a, w, inputType="uint4")
    print(f"  type(result):      {type(result)}")
    print(f"  result repr:       {result!r}")

    attrs = sorted([x for x in dir(result) if not x.startswith("_")])
    print(f"  public attributes: {attrs}")

    if hasattr(result, "numpy"):
        arr = result.numpy()
    else:
        arr = np.asarray(result)
    print(f"  arr shape:  {arr.shape}")
    print(f"  arr dtype:  {arr.dtype}")
    print(f"  arr min/max: {arr.min():.4f} / {arr.max():.4f}")

    # Compare expected: diag([0,1,2,3,4,5,6,7]) @ eye = [0,1,2,3,4,5,6,7]
    expected = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    print(f"  expected:   {expected}")
    print(f"  actual:     {arr}")


def test_consistency(model: Any) -> None:
    """Run the same matmul multiple times to check determinism."""
    print()
    print("=" * 60)
    print("8) Determinism (same input → same output)")
    print("=" * 60)

    a = np.random.randint(0, 16, (1, 1, 16), dtype=np.int32)
    w = np.random.randint(0, 16, (1, 16, 8), dtype=np.int32)

    results = []
    for _ in range(3):
        result = model(a, w, inputType="uint4")
        arr = result.numpy() if hasattr(result, "numpy") else np.asarray(result)
        results.append(arr.copy())

    for i in range(1, len(results)):
        diff = np.abs(results[0].astype(float) - results[i].astype(float))
        print(f"  run 0 vs run {i}: max_diff={diff.max():.6f}, allclose={np.allclose(results[0], results[i])}")


def main() -> None:
    print("Inspecting osimulator Gazelle optical model …")
    print()

    model = load_model()

    inspect_model_object(model)
    test_basic_call(model)
    test_input_types(model)
    test_shapes(model)
    test_value_range(model)
    test_dtype_tolerance(model)
    test_output_properties(model)
    test_consistency(model)

    print()
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

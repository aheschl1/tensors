# tengine

Zero-dependency row‑major N‑dimensional tensor primitives for Rust.

Side project status: APIs may change; scope intentionally small.

## Overview

Minimal building blocks for dense, strided tensors:

* Owned storage (`Tensor<T>`) with shape + stride metadata.
* Zero‑copy immutable and mutable views (`TensorView<'a, T>`, `TensorViewMut<'a, T>`) using an element offset, shape and stride.
* Safe indexing via the `Idx` enum (`Coord`, `At`, `Item`) plus `Index` / `IndexMut` operator for convenience.
* O(1) slicing (`slice(dim, idx)`) producing a view with one fewer dimension.
* Scalar / row / column helpers for ergonomic construction.

The crate does not yet support ranged slicing, broadcasting, or linear algebra. Version `0.x` implies the surface may evolve.

## Installation

Add to `Cargo.toml`:

```toml
tengine = "0.1.0"
```

## Quick Start

```rust
use tengine::ndarray::{Tensor, idx::Idx, MetaTensorView, tensor::{TensorAccess, TensorMut}};

// Matrix 2x3 (row-major):
let m = Tensor::from_buf(vec![1,2,3,4,5,6], vec![2,3]).unwrap();
assert_eq!(m.shape(), &[2,3]);

// Indexing (panic on failure):
assert_eq!(m[vec![0, 1]], 2);

// Safe access via trait:
let v = m.view();
assert_eq!(*v.get(&Idx::Coord(&[1,2])).unwrap(), 6);

// Slice first dimension (row 1): shape becomes [3]
let row1 = v.slice(0, 1).unwrap();
assert_eq!(row1.shape(), &[3]);
assert_eq!(row1[vec![2]], 6);

// Mutation through a mutable view:
let mut m2 = Tensor::row(vec![10, 20, 30]);
let mut mv = m2.view_mut();
mv.set(&Idx::At(1), 99).unwrap();
assert_eq!(m2[vec![0,1]], 99);
```

## Core Types

| Type | Purpose |
|------|---------|
| `Tensor<T>` | Owns a contiguous buffer with row‑major layout. |
| `TensorView<'a, T>` / `TensorViewMut<'a, T>` | Borrowed immutable / mutable views; no allocation. |
| `MetaTensor` | Shape / stride / offset metadata. |
| `Idx<'a>` | Logical index (`Coord(&[usize])`, `At(usize)`, `Item`). |
| `TensorAccess` / `TensorMut` | Traits for read / read+write indexing & slicing. |

## Shapes & Strides

* Shape: lengths per dimension (e.g. `[2, 3, 4]`).
* Stride: element steps per dimension in row‑major order (last dim stride = 1). Computed with `shape_to_stride`.
* Views adjust `offset` + remove a dimension when sliced.

## Error Handling

`TensorError` variants:

* `InvalidShape` – buffer length mismatch or reshape size mismatch.
* `IdxOutOfBounds` – coordinate exceeds dimension length.
* `WrongDims` – index rank mismatch (including `Item` on non‑scalar).
* `InvalidDim` – slice dimension out of range.

Indexing via `[]` unwraps internally and will panic on error; prefer `get`, `get_mut`, or `set` for fallible access.

## Slicing Semantics

`slice(dim, idx)` / `slice_mut(dim, idx)`:

* Removes dimension `dim` from shape & stride vectors.
* Advances offset by `stride[dim] * idx`.
* Produces a view (no copy); scalar result → empty shape `[]`.

## TODO (Future Work)

[ ] Nicer syntax. macro time
[ ] Slicing with ranges
[ ] Elementwise broadcasting
[ ] Basic linear algebra helpers
[ ] Accelerated backends (GPU / parallel)
[ ] x86 SIMD paths
[ ] Nicer indexing syntax sugar

---
Minimal and focused for experimentation; not production hardened.

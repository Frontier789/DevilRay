# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DevilRay is a CUDA-accelerated physically based path tracer (a personal learning project). The author's intent is to write the rendering/math algorithms himself — **do not write or substantially modify the actual rendering algorithms without explicitly being asked to** (path generation, sampling, intersection math, BBH traversal, MIS, etc.). AI assistance is used for tests, glue code, and tooling, and AI-touched files are marked with a disclaimer comment at the top. Tests are fair game; core algorithms are not.

## Build & run

Builds via Conan + CMake presets. A Python venv provides the build tools (conan, cmake, ninja) pinned in `requirements.txt`.

```sh
# one-time per shell: activate the venv that holds conan/cmake/ninja
source .venv/bin/activate

# Release
conan install . --build=missing
cmake --preset conan-release
cmake --build --preset conan-release
build/Release/renderer/devil_ray_renderer      # main interactive renderer
build/Release/bbox_viewer/...                   # BBH/benchmark visualizer

# Debug (note the -s build_type=Debug on conan install)
conan install . --build=missing -s build_type=Debug
cmake --preset conan-debug
cmake --build --preset conan-debug
gdb build/Debug/renderer/devil_ray_renderer
```

`CMAKE_CUDA_ARCHITECTURES` is fixed to **89** (Ada / RTX 40-series) in the top-level `CMakeLists.txt`; change it there if building for other GPUs. CMake options `BUILD_RENDERER` and `BUILD_BBOX_VIEWER` (both ON) gate the two GUI apps.

## Tests

GoogleTest, fetched via `FetchContent`, registered with `gtest_discover_tests`. Run with CTest:

```sh
ctest --preset conan-release --output-on-failure          # all tests
ctest --preset conan-release -R test_intersection -V      # one test target by name
build/Release/test/test_render --gtest_filter='Suite.Case'  # one case directly
```

Test targets live in `test/src/` (`test_alias`, `test_render`, `test_intersection`, `test_mesh`, `test_matrix`) and are added via the `add_test_dr` helper in `test/CMakeLists.txt`. Tests run with the working directory set to the repo root, so reference-image/asset paths in tests are repo-relative. `test_render` compares output against reference images.

## Architecture

Three CMake targets:
- **`devil_ray_lib`** (`src/`, `include/`) — the core library: scene, models, and the CUDA tracer. Both apps link it.
- **`devil_ray_renderer`** (`renderer/`) — interactive GLFW + OpenGL + ImGui front-end that displays the live-rendering image.
- **`bbox_viewer`** (`bbox_viewer/`) — a separate GUI for visualizing the bounding-box hierarchy and running intersection-cost benchmarks.

### Host/device split

The library is compiled as mixed C++/CUDA. The key idiom is the `HD` macro (`include/Utils.hpp`): it expands to `__host__ __device__` under `nvcc` and to nothing otherwise, so the same tracing functions are written once and run on both. Most ray-tracing logic lives in **headers** under `include/tracing/` (e.g. `PathGeneration.hpp`, `IntersectionTestsImpl.hpp`) precisely so it can be included into the `.cu` translation units.

The `__host__` realization of functions are used for testing to ensure correct behaviour without needing the complexity of GPU code in the tests.

The actual GPU kernel and its launch live in `src/device/RendererImpl.hpp` + `src/device/implementation.cu`: `cuda_render` is a 2D kernel (16×16 blocks) with one thread per pixel calling `sampleColor(...)`. `Renderer::schedule_cpu_render()` is intentionally unimplemented (throws) — rendering is GPU-only.

### Device memory model

Two custom containers handle host↔device transfer; never `cudaMalloc` directly in new code, use these:
- **`DeviceVector<T>`** (`include/device/Vector.hpp`) — owns a `std::vector<T>` host copy plus a lazily-allocated device buffer. `push_back` etc. flag `m_deviceNeedsUpdate`; `ensureDeviceAllocation()` (re)allocates and copies to device. `Scene` uses this for `objects` and `materials`; `CudaRandomStates` (`include/device/Random.hpp`) wraps one of `curandState` and exposes `devicePtr()`.
- **`DeviceArray<T>`** (`include/device/Array.hpp`) — fixed-size host+device pair, used for per-pixel buffers (see `Buffers`). Move-only (copy is deleted).

Both are **header-only templates**. The raw CUDA calls are isolated behind a tiny non-template shim — `device_array::{deviceAlloc,deviceFree,copyToDevice,copyToHost}` declared in `Array.hpp` and implemented in `src/device/Array.cpp` (a plain `.cpp`, not `.cu` — it only needs `cuda_runtime.h`). Follow this pattern when adding device storage: keep the template in the header, push the `cudaMalloc`/`cudaMemcpy` into the shim TU.

`Renderer::render()` calls `buffers.ensureDeviceAllocation()` and `scene.ensureDeviceAllocation()` before each kernel launch, then copies results back. Wrap CUDA calls with the `CUDA_ERROR_CHECK()` macro (`include/device/DevUtils.hpp`).

### Rendering pipeline

`Renderer` (`include/Renderer.hpp`) is the orchestrator. Its setters (`setScene`, `setCamera`, `setPixelSampling`, …) are mutex-guarded because the renderer runs on a background thread (`Application::renderWorker` in `renderer/src/RenderThread.cpp`) while the GUI thread reads the latest frame. `needsToBeCleared` triggers buffer/accumulation resets when inputs change.

The path tracer (`include/tracing/PathGeneration.hpp`) accumulates progressive samples into a `Vec4` accumulation buffer (`.w` counts samples). It implements:
- A `PathSampler` that walks up to `max_depth` (10) bounces, building a `PathEntry` array.
- **MIS** (multiple importance sampling) combining BSDF sampling and **NEE** (next-event estimation / direct light sampling) via `powerHeuristic`. Light sources are picked with an `AliasTable` (`DistributionSamplers.hpp`) weighted by radiant power, then a triangle within the mesh, then a point on the triangle.
- Materials are a `std::variant<TransparentMaterial, DiffuseMaterial>` (`tracing/Material.hpp`); dispatch is via `std::get_if`. Transparent materials track a nested-medium IOR `Stack` for refraction (Fresnel/Schlick).
- `DebugOptions` (UVChecker, BariCoords, WindingOrder, …) short-circuit the full integrator to visualize intersection data.

### Geometry & acceleration

`Mesh` (host) → `GpuTris` (device-resident triangle data, owned in `Scene::mesh_storage` as a `std::list` so device pointers stay stable) → `TriangleMesh` (the GPU view: raw `points`/`normals`/`triangles` pointers, `modelToWorld` `Transform`, material index, per-triangle alias sampler, and a `BBHGpuView`). A `TriangleMesh` holds **pointers into** `GpuTris`/`BBH` storage, so that backing storage must outlive it.

`BBH` (`include/models/BBH.hpp`, `src/models/BBH.cpp`) is the bounding-box hierarchy. Nodes carry `left_child`/`right_child` plus a `parent_index`. `generateSimpleBBH` builds it; `createBBHGpuView` exposes nodes to the GPU; `getBoxesOnDepth` feeds the bbox_viewer.

Intersection cost can be instrumented via the `Benchmark` concept (`include/tracing/Benchmark.hpp`): pass `benchmark::HitTests` to count triangle/bbox tests, or `benchmark::Skip` (zero-overhead) in production. `getIntersectionBenchmark` is the instrumented variant of `getIntersection`.

## Conventions

- C++20 and CUDA C++20 throughout. `--expt-relaxed-constexpr` is enabled so `constexpr` helpers work in device code; mark shared math `HD` and prefer `inline HD` free functions in headers.
- Vector/matrix math: `Vec3`/`Vec4`/`Vec2f`, `Matrix`, `Transform`, `AABB` — defined in `include/` (`Utils.hpp`, `models/Matrix.hpp`, `Transform.hpp`). Reuse these rather than introducing new math types.
- Kernel compile diagnostics: the build passes `-Xptxas -v` so register/spill usage prints during compilation — watch it when editing kernels. `devil_ray_lib` is built with `-Wmissing-field-initializers`, so brace-initialize every field of the plain structs passed to kernels (e.g. `Intersection`, `PathEntry`) or expect a warning.
- Captures saved by the renderer land in `captures/` (gitignored); `imgui.ini` is runtime UI state (gitignored).

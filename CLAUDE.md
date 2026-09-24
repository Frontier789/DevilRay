# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DevilRay is a CUDA-accelerated physically based path tracer (a personal learning project).

The author's intent is to write the rendering/math algorithms himself — **do not write or
substantially modify the actual rendering algorithms without explicitly being asked to**
(path generation, sampling, intersection math, BBH traversal, MIS, etc.). AI assistance is
used for tests, glue code, and tooling. Tests are fair game; core algorithms are not.

Mark any file you create or substantially rewrite with a short disclaimer comment at the
top noting it is AI-generated, matching the wording already used in `test/src/`.

## Build & run

Conan + CMake presets. A Python venv provides the pinned build tools (conan, cmake, ninja)
from `requirements.txt`. Presets are named `conan-release` / `conan-debug`.

```sh
source .venv/bin/activate                      # once per shell: conan/cmake/ninja

conan install . --build=missing                 # add -s build_type=Debug for Debug
cmake --preset conan-release
cmake --build --preset conan-release
```

Build artifacts land in `build/<Config>/<subdir>/<exe>`, e.g.
`build/Release/renderer/devil_ray_renderer`.

Notes:

- `CMAKE_CUDA_ARCHITECTURES` is `native`, so the build targets the GPU in the machine.
  Set it explicitly to cross-compile for another card.
- The top-level `CMakeLists.txt` resolves the CUDA toolchain before enabling the CUDA
  language: it prefers the `nvcc` on `PATH` (CMake's own search can find an older
  distro-packaged one) and pins `CMAKE_CUDA_HOST_COMPILER`, because nvcc hard-errors on
  host compilers newer than it supports. Override either with
  `-DCMAKE_CUDA_COMPILER=...` / `-DCMAKE_CUDA_HOST_COMPILER=...`.
- `BUILD_*` options gate each app (all default ON); `devil_ray_lib` and the tests always build.
- Building CUDA sources is slow, and CMake reuses cached objects aggressively. Never read a
  passing test run as proof unless the build that produced it actually recompiled — check
  the build output rather than assuming.

## Tests

GoogleTest via `FetchContent`. Sources live in `test/src/`, one target per area, registered
in `test/CMakeLists.txt` through two helpers:

- `add_test_dr(<name>)` — CPU-only unit tests; `gtest_discover_tests` registers each case
  as its own CTest entry.
- `add_shared_gpu_test_dr(<name> <source>)` — registers the whole binary as a single CTest
  entry so tests touching the CUDA runtime share one context initialization instead of
  paying it per case. Use this for anything that launches kernels or actually allocates
  device memory. Note the device containers allocate lazily, so their host-only behaviour
  can still be tested per-case (see `test_device.cpp`).

```sh
ctest --preset conan-release --output-on-failure           # everything
ctest --preset conan-release -R <target> -V                # one target
build/Release/test/<target> --gtest_filter='Suite.Case'    # one case
```

Tests run with the working directory set to the repo root, so asset and reference-image
paths inside tests are repo-relative.

When adding tests:

- Most tracing code is `HD` (see below), so the host build of a function can be tested
  directly on the CPU without any GPU scaffolding. Prefer that.
- `test/src/TracingTestHelpers.hpp` has the shared fixtures: `Vec3`/`Vec4` near-comparisons,
  a deterministic RNG, a scripted RNG that replays fixed values, small hand-built meshes,
  and `HostObject`, which owns a mesh plus its BBH and hands out a `TriangleMesh` view
  whose pointers reference host memory.
- A CTest entry reported as **Not Run** means its binary failed to build — CTest still
  prints a pass percentage for the rest, so a broken target can hide in a green-looking
  summary. Treat the build log as the source of truth.

## Architecture

### Targets

- **`devil_ray_lib`** (`src/`, `include/`) — the core library: scene, models, tracing, and
  the CUDA kernels. Everything else links it.
- **`devil_ray_renderer`** (`renderer/`) — interactive GLFW + OpenGL + ImGui front-end
  showing the progressive render.
- **`devil_ray_bbox_viewer`** (`bbox_viewer/`) — GUI for visualizing the bounding-box
  hierarchy and running intersection-cost benchmarks.
- **`devil_ray_benchmark`** (`benchmark/`) — headless CLI that casts random rays at a mesh
  and reports triangle/bbox test counts.
- **`sample_image`** (`sample_image/`) — headless image sampler (scaffold).

### Host/device split

The library is mixed C++/CUDA. The `HD` macro (`include/Utils.hpp`) expands to
`__host__ __device__` under nvcc and to nothing otherwise, so tracing functions are written
once and run on both. Most tracing logic therefore lives in **headers** under
`include/tracing/` (`PathGeneration.hpp`, `IntersectionImpl.hpp`, `ShadingUtils.hpp`, …) so
it can be pulled into `.cu` translation units — and so the host realization stays available
to the tests.

### Device memory

Three container templates wrap host↔device transfer. Never call `cudaMalloc` directly in
new code.

- **`DeviceVector<T>`** (`include/device/Vector.hpp`) — growable host `std::vector` plus a
  device copy. Mutations set a dirty flag; `ensureDeviceAllocation()` (re)allocates and
  uploads. Used by `Scene` for `objects` and `materials`.
- **`DeviceArray<T>`** (`include/device/Array.hpp`) — fixed-size host+device pair that keeps
  a host mirror, with `updateDeviceData()` / `updateHostData()` and a `reset()` to the
  initial value. Used for buffers that are read back to the CPU (see `Buffers`).
- **`DeviceBuffer<T>`** (`include/device/Buffer.hpp`) — device-only storage with no
  persistent host mirror; built via `allocate()` / `fromHost()` and read back on demand
  with `toHost()`. Used for the per-path wavefront state.

All three are header-only templates over **`DeviceMemoryManager`**
(`include/device/DeviceMemory.hpp`, implemented in `src/device/DeviceMemory.cu`), an RAII
handle for one device allocation. That is the single translation unit holding raw
`cudaMalloc`/`cudaMemcpy` calls — keep it that way when adding device storage: template in
the header, raw CUDA in the shim.

Wrap CUDA calls with `CUDA_ERROR_CHECK()` (`include/device/DevUtils.hpp`).

### Render loop

`Renderer` (`include/Renderer.hpp`) is the orchestrator. Its setters are mutex-guarded
because rendering runs on a background thread (`Application::renderWorker` in
`renderer/src/RenderThread.cpp`) while the GUI thread reads the latest frame; a
needs-clearing flag resets accumulation when inputs change. `scheduleCpuRender()` is
intentionally unimplemented (throws) — rendering is GPU-only.

Tracing is a **wavefront** path tracer, not a megakernel: `scheduleDeviceRender()`
(`src/device/RendererImpl.hpp`) launches a sequence of 1D kernels over one thread per path,
looping to `Buffers::maxPathLength`:

`initPaths` → `initCameraRays` → per depth { binning → `extendPaths` → `shade` →
`sampleBsdfDirection` } → `accumulateSampleCount`

Per-path state is **structure-of-arrays** across `DeviceBuffer`s — `PathVertexData`
(`include/tracing/Path.hpp`: `t`, throughput, previous BSDF pdf, specular flag, hit ids,
alive flag) and `WavefrontData` (`include/tracing/Wavefront.hpp`: rays, current material,
sort index). Each struct has a mirrored `...Device` plain-pointer view that is what actually
gets passed to kernels.

Between bounces, `DeviceBinning` (`include/device/Binning.hpp`, `src/device/Binning.cu`)
uses CUB stream compaction to gather still-alive paths into `sort_index`, so later kernels
skip dead paths.

Radiance sums into a `Vec4` accumulation buffer whose `.w` counts samples; a separate
kernel bumps that counter exactly once per frame. The integrator combines BSDF sampling
with **NEE** (next-event estimation) through `powerHeuristic` (**MIS**). Lights are chosen
with an `AliasTable` (`include/tracing/DistributionSamplers.hpp`) weighted by radiant power,
then a triangle within the mesh, then a point on that triangle. Materials are a
`std::variant` (`include/tracing/Material.hpp`) dispatched with `std::get_if`. Setting
`DebugOptions` to anything but `Off` takes a separate short path — one `extendPaths` plus
`debugShade` — to visualize intersection data instead of integrating.

### Geometry & acceleration

`Mesh` (host, `include/models/Mesh.hpp`) → `GpuTris` (device-resident triangle data, owned
by `Scene::mesh_storage` as a `std::list` so device pointers stay stable) → `TriangleMesh`
(the GPU view: raw `points`/`normals`/`triangles` pointers, `model_to_world` `Transform`,
material index, per-triangle alias sampler, and a `BBHGpuView`). A `TriangleMesh` holds
**pointers into** that storage, which must outlive it.

`BBH` (`include/models/BBH.hpp`, `src/models/BBH.cpp`) is the bounding-box hierarchy; nodes
carry `left_child`/`right_child` plus `parent_index`. `generateSimpleBBH` builds it — note
it **reorders the mesh's triangles**, so anything derived from triangle order (such as an
area-weighted sampler) must be built afterwards. `createBBHGpuView` exposes nodes to the
GPU; `getBoxesOnDepth` feeds the bbox viewer.

Intersection entry points are declared in `include/tracing/Intersection.hpp`
(`intersectTriangle`, `intersectMesh`, `intersectScene`, `occludedScene`) and return small
hit structs rather than optionals. Cost can be instrumented via the `Benchmark` concept
(`include/tracing/Benchmark.hpp`): the `...Benchmark` variants take `benchmark::HitTests` to
count triangle/bbox tests, or `benchmark::Skip` for zero overhead in production.

Shading normals come from `surfaceNormal` (`include/models/MeshUtils.hpp`), which
interpolates the vertex normals with `barycentricCoordinates`, transforms by the inverse
scale, and flips toward the incoming ray. It expects the hit point in **world** space.
`triangleFaceNormal` is deliberately **not** normalized — its magnitude is twice the
triangle area, which `barycentricCoordinates` relies on.

## Conventions

- C++20 and CUDA C++20 throughout. `--expt-relaxed-constexpr` is enabled so `constexpr`
  helpers work in device code; mark shared math `HD` and prefer `inline HD` free functions
  in headers.
- Reuse the existing math types — `Vec3`/`Vec4`/`Vec2f`/`Size2i`, `Matrix`, `Transform`,
  `AABB` (`include/Utils.hpp`, `include/models/Matrix.hpp`, `include/Transform.hpp`) —
  rather than introducing new ones.
- `devil_ray_lib` builds with `-Wmissing-field-initializers`: brace-initialize *every*
  field of the plain structs handed to kernels, or expect a warning.
- The build passes `-Xptxas -v` and `-lineinfo`, so register and spill counts print while
  compiling — watch them when editing kernels.
- Captures saved by the renderer land in `captures/`; `imgui.ini` holds runtime UI state.
  Both are gitignored.

#pragma once

#include "device/DevUtils.hpp"

#include "tracing/OutputOptions.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Scene.hpp"

#include "Buffers.hpp"
#include "DebugOptions.hpp"

#include <filesystem>
#include <atomic>
#include <vector>
#include <mutex>
#include <cstring>

class Renderer
{
    Buffers buffers;
    PixelSampling pixel_sampling;

    std::vector<uint32_t> pixels;
    std::vector<uint32_t> displayPixels;
    Size2i resolution;
    DebugOptions debug;

    Camera camera;
    Scene scene;
    Timer timer;

    CudaRandomStates cuda_randoms;

    OutputOptions output_options;

    std::mutex renderMutex;
    std::mutex displayMutex;
    bool clearRequested = false;

public:
    Renderer(Size2i resolution);

    const Buffers &getBuffers() const { return buffers; }
    void setDebug(DebugOptions dbg) { debug = std::move(dbg); }

    void setCamera(Camera cam);
    void setScene(Scene scn) { scene = std::move(scn); calculateLightWeights(); }
    void setPixelSampling(PixelSampling sampling) { pixel_sampling = sampling; }
    void setOutputOptions(OutputOptions options) { output_options = std::move(options); }

    void render();
    void clear();

    void createPixels();
    void saveImage(const std::filesystem::path &path);

    const uint32_t *getPixels();
    const Vec4 *getRawPixels();

    void calculateLightWeights();

private:
    void schedule_cpu_render();
    void schedule_device_render();
};
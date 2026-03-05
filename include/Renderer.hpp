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
#include "RunningAverage.hpp"

#include <filesystem>
#include <atomic>
#include <vector>
#include <mutex>
#include <cstring>

class Renderer
{
    Buffers buffers;

    Camera camera;
    DebugOptions debug;
    OutputOptions output_options;
    PixelSampling pixel_sampling;
    Size2i resolution;

    std::vector<uint32_t> pixels;
    std::vector<uint32_t> displayPixels;

    Scene scene;
    
    CudaRandomStates cuda_randoms;
    RunningAverage renderTimes;

    std::mutex renderMutex;
    std::mutex displayMutex;
    bool needsToBeCleared = false;

public:
    Renderer(Size2i resolution);

    const Buffers &getBuffers() const { return buffers; }
    float getMeanRenderTimes() const { return renderTimes.mean(); }

    void setDebug(DebugOptions dbg);
    void setCamera(Camera cam);
    void setScene(Scene scn) { scene = std::move(scn); calculateLightWeights(); }
    void setPixelSampling(PixelSampling sampling);
    void setOutputOptions(OutputOptions options);

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
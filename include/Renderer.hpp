#pragma once

#include "device/DevUtils.hpp"

#include "tracing/DistributionSamplers.hpp"
#include "tracing/OutputOptions.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/Camera.hpp"
#include "tracing/TriangleMesh.hpp"
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
    Buffers m_buffers;
    uint64_t m_totalCasts;

    Camera m_camera;
    DebugOptions m_debug;
    OutputOptions m_output_options;
    PixelSampling m_pixel_sampling;
    Size2i m_resolution;
    AliasTable m_light_sampler;

    std::vector<uint32_t> m_pixels;
    std::vector<uint32_t> m_displayPixels;

    Scene m_scene;

    CudaRandomStates m_cuda_randoms;
    RunningAverage m_renderTimes;

    std::mutex m_renderMutex;
    std::mutex m_displayMutex;
    bool m_needsToBeCleared = false;

public:
    Renderer(Size2i resolution);

    const Buffers &getBuffers() const { return m_buffers; }
    float getMeanRenderTimes() const { return m_renderTimes.mean(); }
    uint64_t getTotalCasts() const { return m_totalCasts; }

    void setDebug(DebugOptions dbg);
    void setCamera(Camera cam);
    void setScene(Scene scn) { m_scene = std::move(scn); calculateLightWeights(); }
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
    void scheduleCpuRender();
    void scheduleDeviceRender();
};
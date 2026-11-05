#pragma once

#include "Buffers.hpp"
#include "DebugOptions.hpp"
#include "RunningAverage.hpp"
#include "Utils.hpp"
#include "device/Binning.hpp"
#include "device/Random.hpp"
#include "tracing/Camera.hpp"
#include "tracing/DistributionSamplers.hpp"
#include "tracing/OutputOptions.hpp"
#include "tracing/Path.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/Scene.hpp"
#include "tracing/Wavefront.hpp"

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <utility>
#include <vector>

class Renderer
{
    Buffers m_buffers;

    WavefrontData m_wavefront;
    PathVertexData m_pathVertices;
    DeviceBinning m_binning;

    uint64_t m_totalCasts;

    Camera m_camera{};
    DebugOptions m_debug = DebugOptions::Off;
    OutputOptions m_output_options;
    PixelSampling m_pixel_sampling;
    float m_environment_map_intensity_scale = 1.0f;
    Size2i m_resolution;
    AliasTable m_light_sampler;

    std::vector<uint32_t> m_pixels;
    std::vector<uint32_t> m_displayPixels;

    Scene m_scene;

    CudaRandom m_cuda_randoms;
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
    void setScene(Scene scn) { m_scene = std::move(scn); calculateLightWeights(); clear(); }
    void setPixelSampling(PixelSampling sampling);
    void setEnvironmentMapIntensityScale(float scale);
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
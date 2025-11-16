#pragma once

#include "device/DevUtils.hpp"

#include "tracing/PixelSampling.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Scene.hpp"

#include "Buffers.hpp"

#include <atomic>
#include <vector>

class Renderer
{
    Buffers buffers;
    PixelSampling pixel_sampling;

    std::vector<uint32_t> pixels;
    Size2i resolution;
    bool debug;
    bool useCuda;

    Camera camera;
    Scene scene;
    Timer timer;

    CudaRandomStates cuda_randoms;

public:
    Renderer(Size2i resolution);

    const Buffers &getBuffers() const {return buffers;}
    Size2i getResolution() const {return resolution;}
    void setDebug(bool dbg) { debug = dbg; }
    void useCudaDevice(bool use) { useCuda = use; }

    void setCamera(Camera cam) { camera = std::move(cam); }
    void setScene(Scene scn) { scene = std::move(scn); calculateLightWeights(); }
    void setPixelSampling(PixelSampling sampling) { pixel_sampling = sampling; }

    void render();
    void clear();

    void createPixels();
    void saveImage(const std::string &fileName);

    const uint32_t *getPixels();

    void calculateLightWeights();

private:
    void schedule_cpu_render();
    void schedule_device_render();
};
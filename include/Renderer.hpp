#pragma once

#include "device/DevUtils.hpp"

#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Scene.hpp"

#include <atomic>
#include <vector>

struct RenderStats
{
    std::atomic<int64_t> total_casts;
};

class Renderer
{
    DeviceArray<Vec4> accumulator;
    std::vector<uint32_t> pixels;
    Size2i resolution;
    bool debug;

    RenderStats stats;
    Camera camera;

    Scene scene;

    Timer timer;

    CudaRandomStates cuda_randoms;
public:


    Renderer(Size2i resolution);

    const RenderStats &getStats() const {return stats;}
    Size2i getResolution() const {return resolution;}
    void setDebug(bool dbg) { debug = dbg; }

    void setCamera(Camera cam) { camera = std::move(cam); }
    void setScene(Scene scn) { scene = std::move(scn); }

    void render();
    void clear();

    void createPixels();
    void saveImage(const std::string &fileName);

    const uint32_t *getPixels();

private:
    void schedule_cpu_render();
    void schedule_device_render();
};
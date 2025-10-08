#pragma once

#include "device/DevUtils.hpp"

#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Scene.hpp"

#include <atomic>
#include <vector>

struct Outputs
{
    Outputs(Size2i resolution);
    
    void reset();

    uint64_t totalCasts() const;
    
    DeviceArray<Vec4> color;
    mutable DeviceArray<uint32_t> casts;
};

class Renderer
{
    Outputs outputs;

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

    const Outputs &getOutputs() const {return outputs;}
    Size2i getResolution() const {return resolution;}
    void setDebug(bool dbg) { debug = dbg; }
    void useCudaDevice(bool use) { useCuda = use; }

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
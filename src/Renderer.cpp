#include "Renderer.hpp"
#include "Image.hpp"

#include "tracing/SampleScene.hpp"

Renderer::Renderer(Size2i resolution)
    : accumulator(resolution.width * resolution.height, Vec4{0,0,0,0})
    , pixels(resolution.width * resolution.height)
    , resolution(resolution)
    , cuda_randoms(resolution)
{
    
}

void Renderer::createPixels()
{
    auto toSRGB = [](float linear) -> float
    {
        const float r = std::clamp(linear, 0.f, 1.f);
        if (r < 0.0031308) return r * 12.92;
        return std::pow(r, 1/2.4f)*1.055 - 0.055f;
    };

    auto packPixel = [&](float r, float g, float b, float a) -> uint32_t {
        return (static_cast<uint32_t>(toSRGB(r) * 255)<< 0) |
            (static_cast<uint32_t>(toSRGB(g) * 255)<< 8) |
            (static_cast<uint32_t>(toSRGB(b) * 255)<<16) |
            (static_cast<uint32_t>(std::clamp(a, 0.0f, 1.0f) * 255)<<24);
    };

    auto setPixel = [&](int x, int y, float r, float g, float b, float a=1.0f) {
        const auto flipped_y = resolution.height - y - 1;
        pixels[x + flipped_y*resolution.width] = packPixel(r,g,b,a);
    };

    accumulator.updateHostData();

    const auto data = accumulator.hostPtr();

    for (int y=0;y<resolution.height;++y)
    for (int x=0;x<resolution.width;++x)
    {
        const auto pix = data[x + y*resolution.width];
        if (pix.w == 0)
            setPixel(x, y, 0,0,0);
        else
            setPixel(x, y, pix.x / pix.w, pix.y / pix.w, pix.z / pix.w);
    }
}

void Renderer::saveImage(const std::string &fileName)
{
    createPixels();

    savePNG(fileName, pixels, resolution);
}

const uint32_t *Renderer::getPixels()
{
    createPixels();

    return pixels.data();
}

void Renderer::clear()
{
    accumulator.reset();
}

void Renderer::schedule_cpu_render()
{
    const auto objects = std::span{scene.objects.hostPtr(), scene.objects.size()};
    const auto materials = std::span{scene.materials.hostPtr(), scene.materials.size()};

    parallel_for(resolution.height, [&](int y){

        Random random = RandomPool::singleton().borrowRandom();
        //std::cout << "Got random id " << random.get_id() << std::endl;

        const int max_depth = debug ? 1 : 5;

        int ray_casts = 0;

        for (int x=0;x<resolution.width;++x)
        {
            const auto iterations = 10;
            
            auto &pix = accumulator.hostPtr()[x + y*resolution.width];
            pix.w += iterations;

            const auto ray = cameraRay(camera, Vec2{x, y});
            const auto sample = sampleColor(ray, max_depth, objects, materials, debug, iterations, random);

            pix = pix + sample.color;
            ray_casts += sample.casts;
        }
        RandomPool::singleton().returnRandom(std::move(random));
        stats.total_casts += ray_casts;

    });
}

void Renderer::render()
{
    accumulator.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();
    
    scene.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();

    schedule_device_render();   
}

#include "Renderer.hpp"
#include "Image.hpp"

#include "tracing/SampleScene.hpp"


Outputs::Outputs(Size2i resolution)
    : cameraPaths(resolution.area(), {})
    , color(resolution.area(), Vec4{0,0,0,0})
    , casts(resolution.area(), 0)
{

}

void Outputs::reset()
{
    color.reset();
    casts.reset();
}

uint64_t Outputs::totalCasts() const
{
    const auto ptr = casts.hostPtr();
    const auto total = std::accumulate(ptr, ptr + casts.size(), uint64_t{0});

    return total;
}

Renderer::Renderer(Size2i resolution)
    : outputs(resolution)
    , pixel_sampling(PixelSampling::UniformRandom)
    , pixels(resolution.area())
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

    if (useCuda) {
        outputs.color.updateHostData();
    }

    const auto data = outputs.color.hostPtr();

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
    outputs.reset();
}

void Renderer::schedule_cpu_render()
{
    const auto objects = std::span{scene.objects.hostPtr(), scene.objects.size()};
    const auto materials = std::span{scene.materials.hostPtr(), scene.materials.size()};

    parallel_for(resolution.height, [&](int y){

        Random random = RandomPool::singleton().borrowRandom();
        //std::cout << "Got random id " << random.get_id() << std::endl;

        for (int x=0;x<resolution.width;++x)
        {
            const auto iterations = debug ? 1 : 10;
            
            const auto idx = x + y*resolution.width;
            auto &pix = outputs.color.hostPtr()[idx];
            auto *path = outputs.cameraPaths.hostPtr()[idx].data();

            SampleStats stats{.ray_casts = 0};
            sampleColor(Vec2{x, y}, pix, stats, camera, pixel_sampling, objects, materials, path, debug, random);

            outputs.casts.hostPtr()[idx] += stats.ray_casts;
        }
        RandomPool::singleton().returnRandom(std::move(random));

    });
}

void Renderer::render()
{
    if (!useCuda) {
        schedule_cpu_render();
        return;
    }

    outputs.cameraPaths.ensureDeviceAllocation();
    outputs.color.ensureDeviceAllocation();
    outputs.casts.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();

    scene.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();

    schedule_device_render();   
}

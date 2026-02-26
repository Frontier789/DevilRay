#include "Renderer.hpp"
#include "Image.hpp"

#include "tracing/PathGeneration.hpp"
#include "tracing/LightSampling.hpp"
#include <iostream>


Buffers::Buffers(Size2i resolution)
    : lightWeights(0, 0)
    , color(resolution.area(), Vec4{0,0,0,0})
    , casts(resolution.area(), 0)
{

}

void Buffers::reset()
{
    color.reset();
    casts.reset();
}

void Buffers::ensureDeviceAllocation()
{
    lightWeights.ensureDeviceAllocation();
    color.ensureDeviceAllocation();
    casts.ensureDeviceAllocation();
}

uint64_t Buffers::totalCasts() const
{
    const auto ptr = casts.hostPtr();
    const auto total = std::accumulate(ptr, ptr + casts.size(), uint64_t{0});

    return total;
}

Renderer::Renderer(Size2i resolution)
    : buffers(resolution)
    , pixel_sampling(PixelSampling::UniformRandom)
    , pixels(resolution.area())
    , resolution(resolution)
    , cuda_randoms(resolution)
    , output_options(OutputOptions{.linearity = OutputLinearity::GammaCorrected})
{
    
}

void Renderer::createPixels()
{
    const auto linearity = output_options.linearity;
    auto toSRGB = [&linearity](float linear) -> float
    {
        if (linearity == OutputLinearity::Linear) return linear;

        const float r = std::clamp(linear, 0.f, 1.f);
        if (r < 0.0031308) return r * 12.92;
        return std::pow(r, 1/2.4f)*1.055 - 0.055f;
    };

    auto to_u32 = [](auto f) -> uint32_t {return static_cast<uint32_t>(std::round(f));};

    auto packPixel = [&](float r, float g, float b, float a) -> uint32_t {
        return (to_u32(toSRGB(r) * 255)<< 0) |
            (to_u32(toSRGB(g) * 255)<< 8) |
            (to_u32(toSRGB(b) * 255)<<16) |
            (to_u32(std::clamp(a, 0.0f, 1.0f) * 255)<<24);
    };

    auto setPixel = [&](int x, int y, float r, float g, float b, float a=1.0f) {
        const auto flipped_y = resolution.height - y - 1;
        pixels[x + flipped_y*resolution.width] = packPixel(r,g,b,a);
    };

    if (useCuda) {
        buffers.color.updateHostData();
    }

    const auto data = buffers.color.hostPtr();

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

void Renderer::saveImage(const std::filesystem::path &path)
{
    createPixels();

    savePNG(path.string(), pixels, resolution);
}

const uint32_t *Renderer::getPixels()
{
    createPixels();

    return pixels.data();
}

const Vec4 *Renderer::getRawPixels()
{
    if (useCuda) {
        buffers.color.updateHostData();
    }

    return buffers.color.hostPtr();
}


namespace
{
    inline HD float rgbLuminance(const Vec4 &rgb)
    {
        return 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
    }
}

void Renderer::calculateLightWeights()
{
    const auto materials = scene.materials.hostPtr();
    const auto objects = scene.objects.hostPtr();
    const auto N = scene.objects.size();

    buffers.lightWeights = DeviceArray<float>(N, 0);
    const auto weights = buffers.lightWeights.hostPtr();

    float weightSum = 0.0f;

    for (size_t i=0;i<N;++i)
    {
        const auto matIndex = getMaterial(objects[i]);

        const auto A = surfaceArea(objects[i]);
        const auto M = radiantExitance(materials[matIndex]);

        weights[i] = A*rgbLuminance(M);
        weightSum += weights[i];
    }

    for (size_t i=0;i<N;++i)
    {
        weights[i] /= weightSum;

        std::cout << "Object #" << i << ": " << weights[i] << std::endl;
    }
}

void Renderer::clear()
{
    buffers.reset();
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
            const auto idx = x + y*resolution.width;
            auto &pix = buffers.color.hostPtr()[idx];

            SampleStats stats{.ray_casts = 0};
            sampleColor(Vec2{x, y}, pix, stats, camera, pixel_sampling, objects, materials, debug, random);

            buffers.casts.hostPtr()[idx] += stats.ray_casts;
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

    buffers.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();

    scene.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();

    schedule_device_render();   
}

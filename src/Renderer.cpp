#include "Renderer.hpp"
#include "Image.hpp"

#include "tracing/PathGeneration.hpp"
#include "tracing/LightSampling.hpp"
#include <iostream>


Buffers::Buffers(Size2i resolution)
    : color(resolution.area(), Vec4{0,0,0,0})
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
    , totalCasts(0)
    , pixel_sampling(PixelSampling::UniformRandom)
    , pixels(resolution.area())
    , displayPixels(resolution.area(), 0)
    , resolution(resolution)
    , cuda_randoms(resolution)
    , output_options(OutputOptions{.linearity = OutputLinearity::GammaCorrected})
    , renderTimes(20)
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

    buffers.color.updateHostData();

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
    std::scoped_lock guard{displayMutex};

    savePNG(path.string(), displayPixels, resolution);
}

const uint32_t *Renderer::getPixels()
{
    std::scoped_lock guard{displayMutex};

    return displayPixels.data();
}

const Vec4 *Renderer::getRawPixels()
{
    buffers.color.updateHostData();

    return buffers.color.hostPtr();
}

void Renderer::calculateLightWeights()
{
    const auto materials = scene.materials.hostPtr();
    const auto objects = scene.objects.hostPtr();
    const auto N = scene.objects.size();

    std::vector<float> weights(N, 0);

    float total_radiant_power = 0.0f;

    for (size_t i=0;i<N;++i)
    {
        const auto matIndex = objects[i].material;

        const auto A = objects[i].surface_area;
        const auto M = radiantExitance(materials[matIndex]);

        weights[i] = A*luminance(M);
        total_radiant_power += weights[i];
    }

    std::cout << "Generating AliasTable from weights:\n";
    for (size_t i=0;i<N;++i)
    {
        weights[i] /= total_radiant_power;

        std::cout << "  Object #" << i << ": " << weights[i] << std::endl;
    }

    light_sampler = generateAliasTable(weights);
    light_sampler.entries.ensureDeviceAllocation();

    scene.info.total_radiant_power = total_radiant_power;
    std::cout << "Total radiant power: " << total_radiant_power << std::endl;

    for (const auto e : std::span{light_sampler.entries.hostPtr(), light_sampler.entries.size()})
    {
        std::cout << "  A=" << e.A << ", B=" << e.B << " p_A=" << e.p_A << " pdf_A=" << e.pdf_A << " pdf_B=" << e.pdf_B << std::endl;
    }
}


void Renderer::setDebug(DebugOptions dbg)
{
    std::scoped_lock guard{renderMutex};

    debug = std::move(dbg);
    needsToBeCleared = true;
}

void Renderer::setCamera(Camera cam)
{
    std::scoped_lock guard{renderMutex};

    camera = std::move(cam);
    needsToBeCleared = true;
}

void Renderer::clear()
{
    std::scoped_lock guard{renderMutex};

    needsToBeCleared = true;
}

void Renderer::setPixelSampling(PixelSampling sampling)
{
    std::scoped_lock guard{renderMutex};

    pixel_sampling = sampling;
    needsToBeCleared = true;
}

void Renderer::setOutputOptions(OutputOptions options)
{
    std::scoped_lock guard{renderMutex};

    output_options = std::move(options);
}

void Renderer::schedule_cpu_render()
{
    throw std::runtime_error("unimplemented");
}

void Renderer::render()
{
    {
        std::scoped_lock guard{renderMutex};

        if (needsToBeCleared)
        {
            renderTimes.reset();
            buffers.reset();
            needsToBeCleared = false;
        }
    }

    buffers.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();

    scene.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();


    Timer t;
    schedule_device_render();

    createPixels();

    {
        std::scoped_lock guard{displayMutex};
        std::memcpy(displayPixels.data(), pixels.data(), pixels.size() * sizeof(uint32_t));
    }
    const auto elapsed_ms = t.elapsed_seconds() * 1000;
    renderTimes.add(elapsed_ms);


    buffers.casts.updateHostData();
    totalCasts = buffers.totalCasts();
}

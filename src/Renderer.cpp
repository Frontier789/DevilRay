#include "Renderer.hpp"
#include "Image.hpp"

#include "tracing/PathGeneration.hpp"
#include "tracing/LightSampling.hpp"
#include <iostream>
#include <numeric>


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
    : m_buffers(resolution)
    , m_totalCasts(0)
    , m_output_options(OutputOptions{.linearity = OutputLinearity::GammaCorrected})
    , m_pixel_sampling(PixelSampling::UniformRandom)
    , m_resolution(resolution)
    , m_pixels(resolution.area())
    , m_displayPixels(resolution.area(), 0)
    , m_cuda_randoms(resolution)
    , m_renderTimes(20)
{

}

void Renderer::createPixels()
{
    const auto linearity = m_output_options.linearity;
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
        const auto flipped_y = m_resolution.height - y - 1;
        m_pixels[x + flipped_y*m_resolution.width] = packPixel(r,g,b,a);
    };

    m_buffers.color.updateHostData();

    const auto data = m_buffers.color.hostPtr();

    for (int y=0;y<m_resolution.height;++y)
    for (int x=0;x<m_resolution.width;++x)
    {
        const auto pix = data[x + y*m_resolution.width];
        if (pix.w == 0)
            setPixel(x, y, 0,0,0);
        else
            setPixel(x, y, pix.x / pix.w, pix.y / pix.w, pix.z / pix.w);
    }
}

void Renderer::saveImage(const std::filesystem::path &path)
{
    std::scoped_lock guard{m_displayMutex};

    savePNG(path.string(), m_displayPixels, m_resolution);
}

const uint32_t *Renderer::getPixels()
{
    std::scoped_lock guard{m_displayMutex};

    return m_displayPixels.data();
}

const Vec4 *Renderer::getRawPixels()
{
    m_buffers.color.updateHostData();

    return m_buffers.color.hostPtr();
}

void Renderer::calculateLightWeights()
{
    const auto materials = m_scene.materials.hostPtr();
    const auto objects = m_scene.objects.hostPtr();
    const auto N = m_scene.objects.size();

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

    m_light_sampler = generateAliasTable(weights);
    m_light_sampler.entries.ensureDeviceAllocation();

    m_scene.info.total_radiant_power = total_radiant_power;
    std::cout << "Total radiant power: " << total_radiant_power << std::endl;

    for (const auto e : std::span{m_light_sampler.entries.hostPtr(), m_light_sampler.entries.size()})
    {
        std::cout << "  A=" << e.A << ", B=" << e.B << " p_A=" << e.p_A << " pdf_A=" << e.pdf_A << " pdf_B=" << e.pdf_B << std::endl;
    }
}


void Renderer::setDebug(DebugOptions dbg)
{
    std::scoped_lock guard{m_renderMutex};

    m_debug = std::move(dbg);
    m_needsToBeCleared = true;
}

void Renderer::setCamera(Camera cam)
{
    std::scoped_lock guard{m_renderMutex};

    m_camera = std::move(cam);
    m_needsToBeCleared = true;
}

void Renderer::clear()
{
    std::scoped_lock guard{m_renderMutex};

    m_needsToBeCleared = true;
}

void Renderer::setPixelSampling(PixelSampling sampling)
{
    std::scoped_lock guard{m_renderMutex};

    m_pixel_sampling = sampling;
    m_needsToBeCleared = true;
}

void Renderer::setOutputOptions(OutputOptions options)
{
    std::scoped_lock guard{m_renderMutex};

    m_output_options = std::move(options);
}

void Renderer::scheduleCpuRender()
{
    throw std::runtime_error("unimplemented");
}

void Renderer::render()
{
    {
        std::scoped_lock guard{m_renderMutex};

        if (m_needsToBeCleared)
        {
            m_renderTimes.reset();
            m_buffers.reset();
            m_needsToBeCleared = false;
        }
    }

    m_buffers.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();

    m_scene.ensureDeviceAllocation();
    CUDA_ERROR_CHECK();


    Timer t;
    scheduleDeviceRender();

    createPixels();

    {
        std::scoped_lock guard{m_displayMutex};
        std::memcpy(m_displayPixels.data(), m_pixels.data(), m_pixels.size() * sizeof(uint32_t));
    }
    const auto elapsed_ms = t.elapsedSeconds() * 1000;
    m_renderTimes.add(elapsed_ms);


    m_buffers.casts.updateHostData();
    m_totalCasts = m_buffers.totalCasts();
}

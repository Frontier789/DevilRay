#include <tracing/DistributionSamplers.hpp>
#include <Image.hpp>

#include <random>

struct CpuRand
{
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{0,1};

    float rnd()
    {
        return dist(rng);
    }
};

template<typename T>
T lerp(const T& start, const T& end, float t) {
    return start * (1.0f - t) + end * t;
}

template<typename T>
void circle(Image<T>& image, Vec2i center, float radius, const T& color) {
    if (radius <= 0.0f) return;

    int minX = std::max(0, static_cast<int>(std::floor(center.x - radius - 1.5f)));
    int maxX = std::min(image.size.width - 1, static_cast<int>(std::ceil(center.x + radius + 1.5f)));
    int minY = std::max(0, static_cast<int>(std::floor(center.y - radius - 1.5f)));
    int maxY = std::min(image.size.height - 1, static_cast<int>(std::ceil(center.y + radius + 1.5f)));

    for (int y = minY; y <= maxY; ++y) {
        float offsetY = y - center.y;
        float offsetYSquared = offsetY * offsetY;

        for (int x = minX; x <= maxX; ++x) {
            float offsetX = x - center.x;
            float distanceFromCenter = std::sqrt(offsetX * offsetX + offsetYSquared);
            float distanceToEdge = distanceFromCenter - radius;

            float blendRatio = 0.0f;
            if (distanceToEdge < -0.5f) {
                blendRatio = 1.0f;
            } else if (distanceToEdge < 0.5f) {
                float progress = distanceToEdge + 0.5f;
                blendRatio = 1.0f - (progress * progress * (3.0f - 2.0f * progress));
            }

            if (blendRatio > 0.0f) {
                Vec2i pixelPos{x, y};
                if (blendRatio >= 1.0f) {
                    image[pixelPos] = color;
                } else {
                    image[pixelPos] = lerp(image[pixelPos], color, blendRatio);
                }
            }
        }
    }
}
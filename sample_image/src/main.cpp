#include <tracing/DistributionSamplers.hpp>
#include <Image.hpp>

#include <iostream>
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

int main()
{
    const auto image_path = "sample_image/lady_small.jpg";

    Timer t;
    const auto image = loadHDR(image_path);

    const auto load_time = t.elapsedSeconds();
    std::cout << "Loaded image sized " << image.size.width << "x" << image.size.height << " in " << load_time*1000 << "ms" << std::endl;

    auto luminance = intensity(image);

    t = Timer{};
    const auto alias_table = generateAliasTable(luminance.view());

    const auto table_time = t.elapsedSeconds();
    std::cout << "Built Alias Table in " << table_time*1000 << "ms" << std::endl;

    CpuRand rng;
    const auto alias_table_view = AliasImageTableView{
        .pixel_entries = alias_table.pixel_entries.hostPtr(),
        .row_entries = alias_table.row_entries.hostPtr(),
        .image_size = alias_table.image_size,
    };

    const auto s = luminance.size;

    auto dots = Image1f::create(s, 0);
    auto closest_id = Image<int>::create(s, -1);

    std::vector<Vec2i> seeds;

    struct Update
    {
        Vec2i p;
        int id;
    };

    std::vector<Update> updates;

    for (int i=0;i<1300;++i) {
        const auto s = sample(alias_table_view, rng);
        seeds.push_back(s.pixel_coordinate);
        updates.push_back(Update{s.pixel_coordinate, i});
        dots[s.pixel_coordinate] = 1;
    }

    while (updates.size() > 0) {
        std::vector<Update> next_updates;
    
        for (const auto u : updates)
        {
            if (u.p.x < 0 || u.p.y < 0 ||
                u.p.x >= s.width || u.p.y >= s.height) continue;

            if (closest_id[u.p] != -1)
            {
                const auto cid = closest_id[u.p];
                const Vec2f cp = seeds[cid];
                const Vec2f np = seeds[u.id];
                
                if ((cp - u.p).length() <= (np - u.p).length())
                    continue;
            }

            closest_id[u.p] = u.id;
            next_updates.push_back(Update{u.p + Vec2i{1,0}, u.id});
            next_updates.push_back(Update{u.p + Vec2i{-1,0}, u.id});
            next_updates.push_back(Update{u.p + Vec2i{0,1}, u.id});
            next_updates.push_back(Update{u.p + Vec2i{0,-1}, u.id});
        }
    
        updates = std::move(next_updates);
    }

    struct Stat
    {
        double sum = 0;
        int pix_count = 0;
    };

    std::vector<Stat> region_stats(seeds.size(), Stat{0,0});
    for (int i=0;i<s.area();++i) {
        const auto p = Vec2i{.x=i%s.width, .y=i/s.width};

        const auto id = closest_id[p];
        if (id >= 0) {
            region_stats[id].sum += luminance[p];
            region_stats[id].pix_count += 1;
        }
    }

    auto img = Image4f::create(s);
    for (int y=0;y<s.height;++y)
    for (int x=0;x<s.width;++x)
    {
        const auto p = Vec2i{x,y};
        const auto id = closest_id[p];

        if (id >= 0) {
            const auto f = static_cast<float>(region_stats[id].sum / region_stats[id].pix_count);
            img[p] = Vec4{f,f,f,1};
        }
    }

    for (const auto p : seeds) {
        circle(img, p, 3, Vec4{0.5,0.6,0.9,1});
    }

    // std::vector<uint32_t> region_colors(seeds.size(), 0);
    // for (auto &c : region_colors) c = Vec4{rng.rnd(),rng.rnd(),rng.rnd(),1}.to8bitColor();

    std::vector<uint32_t> pixel_data(s.area(), 0);

    for (int i=0;i<s.area();++i) {
        const auto p = Vec2i{.x=i%s.width, .y=i/s.width};

        pixel_data[i] = img[p].to8bitColor();
    }
    savePNG("voronoi.png", pixel_data, s);

    for (int i=0;i<s.area();++i) {
        const auto p = Vec2i{.x=i%s.width, .y=i/s.width};

        const auto f = dots[p];
        pixel_data[i] = Vec4{f,f,f,1}.to8bitColor();
    }
    savePNG("sample_points.png", pixel_data, s);

}
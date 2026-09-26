#include <tracing/DistributionSamplers.hpp>
#include <Image.hpp>

#include <tools.hpp>
#include <CommandLine.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace
{
    std::vector<Vec2i> sample_points_on_image(Image1f &luminance, int point_count, CpuRand &rng)
    {
        Timer t;
        const auto alias_table = generateAliasTable(luminance.view());

        const auto table_time = t.elapsedSeconds();
        std::cout << "Built Alias Table in " << table_time*1000 << "ms" << std::endl;

        const auto alias_table_view = AliasImageTableView{
            .pixel_entries = alias_table.pixel_entries.hostPtr(),
            .row_entries = alias_table.row_entries.hostPtr(),
            .image_size = alias_table.image_size,
        };

        std::vector<Vec2i> points;
        points.reserve(point_count);

        for (int i=0;i<point_count;++i) {
            points.push_back(sample(alias_table_view, rng).pixel_coordinate);
        }

        return points;
    }

    Image<int> build_voronoi(const std::vector<Vec2i> &seeds, Size2i size)
    {
        auto closest_id = Image<int>::create(size, -1);

        struct Update
        {
            Vec2i p;
            int id;
        };

        std::vector<Update> updates;
        for (int i=0;i<static_cast<int>(seeds.size());++i) {
            updates.push_back(Update{seeds[i], i});
        }

        while (updates.size() > 0) {
            std::vector<Update> next_updates;

            for (const auto u : updates)
            {
                if (u.p.x < 0 || u.p.y < 0 ||
                    u.p.x >= size.width || u.p.y >= size.height) continue;

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

        return closest_id;
    }

    struct RegionStat
    {
        Vec4 sum = Vec4{0,0,0,0};
        int pix_count = 0;

        Vec4 mean() const { return sum / pix_count; }
    };

    std::vector<RegionStat> voronoi_stats(const Image<int> &closest_id, const Image4f &image, size_t region_count)
    {
        std::vector<RegionStat> region_stats(region_count, RegionStat{0,0});

        for (std::size_t i=0;i<closest_id.pixels.size();++i) {
            const auto id = closest_id.pixels[i];
            if (id >= 0) {
                region_stats[id].sum += image.pixels[i];
                region_stats[id].pix_count += 1;
            }
        }

        return region_stats;
    }

    Image4f voronoi_color(const Image<int> &closest_id, const std::vector<RegionStat> &region_stats)
    {
        auto img = Image4f::create(closest_id.size);

        for (std::size_t i=0;i<closest_id.pixels.size();++i) {
            const auto id = closest_id.pixels[i];
            if (id >= 0) {
                img.pixels[i] = region_stats[id].mean();
            }
        }

        return img;
    }

    constexpr float display_gamma = 2.2f;

    Vec4 linear_to_display(const Vec4 &linear)
    {
        const auto encode = [](float channel) { return std::pow(std::max(channel, 0.0f), 1.0f / display_gamma); };

        return Vec4{encode(linear.x), encode(linear.y), encode(linear.z), linear.w};
    }

    std::vector<uint32_t> image4f_to_u32(const Image4f &image)
    {
        std::vector<uint32_t> pixel_data;
        pixel_data.reserve(image.pixels.size());

        for (const auto &pixel : image.pixels) {
            pixel_data.push_back(linear_to_display(pixel).to8bitColor());
        }

        return pixel_data;
    }
}

int main(int argc, char *argv[])
{
    const auto options = parseCommandLineOrExit(argc, argv);
    
    std::cout << "Loading " << options.image_path << std::endl;
    const auto image = loadHDR(options.image_path);
    auto luminance = intensity(image);

    std::cout << "Sampling image" << std::endl;
    CpuRand rng;
    const auto seeds = sample_points_on_image(luminance, options.point_count, rng);

    std::cout << "Building Voronoi" << std::endl;
    const auto closest_id = build_voronoi(seeds, luminance.size);
    const auto region_stats = voronoi_stats(closest_id, image, static_cast<int>(seeds.size()));

    std::cout << "Coloring Voronoi" << std::endl;
    auto img = voronoi_color(closest_id, region_stats);

    if (options.plot_points) {
        for (const auto p : seeds) {
            circle(img, p, 3, Vec4{0.5,0.8,0.95,1});
        }
    }

    const auto output = "voronoi.png";
    std::cout << "Saving " << output << std::endl;
    savePNG(output, image4f_to_u32(img), img.size);
}

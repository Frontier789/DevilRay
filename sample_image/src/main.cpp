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

int main()
{
    const auto image_path = "/home/matyas/work/DevilRay/sample_image/meadow_2_4k.hdr";

    Timer t;
    const auto image = loadHDR(image_path);

    const auto load_time = t.elapsedSeconds();
    std::cout << "Loaded image sized " << image.size.width << "x" << image.size.height << " in " << load_time*1000 << "ms" << std::endl;

    auto luminance = intensity(image);

    auto lum_view = ImageView1f{
        .pixels = luminance.pixels.data(),
        .size = luminance.size,
    };

    t = Timer{};
    const auto alias_table = generateAliasTable(lum_view);

    const auto table_time = t.elapsedSeconds();
    std::cout << "Built Alias Table in " << table_time*1000 << "ms" << std::endl;

    CpuRand rng;
    const auto alias_table_view = AliasImageTableView{
        .pixel_entries = alias_table.pixel_entries.hostPtr(),
        .row_entries = alias_table.row_entries.hostPtr(),
        .image_size = alias_table.image_size,
    };

    for (int i=0;i<13;++i) {
        const auto s = sample(alias_table_view, rng);

        std::cout << "Sample #" << i << ": " << s.pixel_coordinate.x << "," << s.pixel_coordinate.y << std::endl;
    }
}
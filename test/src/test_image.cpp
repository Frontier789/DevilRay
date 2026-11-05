// AI-generated tests (Claude), reviewed by hand before committing.

#include "Image.hpp"
#include "ImageMetadata.hpp"
#include "Utils.hpp"
#include "VectorAssertions.hpp"

#include <exiv2/image.hpp>
#include <exiv2/properties.hpp>
#include <exiv2/tags.hpp>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
    Vec2i pixelAt(int x, int y)
    {
        return Vec2i{.x = x, .y = y};
    }

    std::filesystem::path outputFolder()
    {
        const auto folder = std::filesystem::path("test_output/test_image");
        std::filesystem::create_directories(folder);

        return folder;
    }

    std::string savedPngWithOnePixelPerColor(const std::string &file_name, const std::vector<Vec4> &colors)
    {
        const auto path = (outputFolder() / file_name).string();

        std::vector<uint32_t> packed_pixels;
        for (const auto &color : colors) packed_pixels.push_back(color.to8bitColor());

        savePNG(path, packed_pixels, Size2i{static_cast<int>(colors.size()), 1});

        return path;
    }

    std::optional<std::string> xmpValue(const std::string &path, const std::string &key)
    {
        auto image = Exiv2::ImageFactory::open(path);
        image->readMetadata();

        const auto found = image->xmpData().findKey(Exiv2::XmpKey(key));
        if (found == image->xmpData().end()) return std::nullopt;

        return found->toString();
    }

    bool hasExifTag(const std::string &path, const std::string &key)
    {
        auto image = Exiv2::ImageFactory::open(path);
        image->readMetadata();

        return image->exifData().findKey(Exiv2::ExifKey(key)) != image->exifData().end();
    }
}

// --- Image<T> ---

namespace
{
    template<typename T>
    std::vector<T> pixelsOf(const Image<T> &image)
    {
        const auto span = image.pixels.hostSpan();
        return std::vector<T>(span.begin(), span.end());
    }
}

TEST(ImageTest, CreateFillsEveryPixelWithTheGivenValue)
{
    const auto image = Image1f::create(Size2i{3, 2}, 0.5f);

    EXPECT_EQ(image.size.width, 3);
    EXPECT_EQ(image.size.height, 2);
    EXPECT_EQ(pixelsOf(image), std::vector<float>(6, 0.5f));
}

TEST(ImageTest, CreateWithoutValueZeroInitializesPixels)
{
    const auto scalar_image = Image1f::create(Size2i{2, 2});
    EXPECT_EQ(pixelsOf(scalar_image), std::vector<float>(4, 0.0f));

    const auto color_image = Image4f::create(Size2i{2, 1});
    for (const auto &pixel : pixelsOf(color_image))
        test::expectVec4Near(pixel, {0, 0, 0, 0});
}

TEST(ImageTest, CreateUninitializedHasTheRequestedSizeAndKeepsWrittenPixels)
{
    auto image = Image4f::create_uninitialized(Size2i{3, 2});

    EXPECT_EQ(image.size.width, 3);
    EXPECT_EQ(image.size.height, 2);
    ASSERT_EQ(image.pixels.size(), 6u);

    for (int i = 0; i < 6; ++i)
        image.pixels.hostSpan()[i] = Vec4{static_cast<float>(i), 0, 0, 1};

    for (int i = 0; i < 6; ++i)
        test::expectVec4Near(image.pixels.hostSpan()[i], {static_cast<float>(i), 0, 0, 1});
}

TEST(ImageTest, IndexingIsRowMajorWithXAsColumn)
{
    auto image = Image1f::create(Size2i{3, 2});

    image[pixelAt(2, 1)] = 7.0f;

    constexpr int index_of_last_column_in_second_row = 2 + 3 * 1;
    EXPECT_EQ(image.pixels.hostSpan()[index_of_last_column_in_second_row], 7.0f);
}

TEST(ImageTest, ConstIndexingReadsTheSamePixelAsMutableIndexing)
{
    auto image = Image1f::create(Size2i{3, 2});
    image[pixelAt(1, 1)] = 4.0f;

    const auto &const_image = image;

    EXPECT_EQ(const_image[pixelAt(1, 1)], 4.0f);
}

TEST(ImageTest, HostViewSharesPixelsAndSizeWithTheImage)
{
    auto image = Image1f::create(Size2i{4, 2});

    const ImageView1f view = image.hostView();
    view.pixels[5] = 3.0f;

    EXPECT_EQ(view.pixels, image.pixels.hostPtr());
    EXPECT_EQ(view.size.width, 4);
    EXPECT_EQ(view.size.height, 2);
    EXPECT_EQ(image.pixels.hostSpan()[5], 3.0f);
}

TEST(ImageTest, ConstImageGivesAReadOnlyHostViewOverTheSamePixels)
{
    const auto image = Image1f::create(Size2i{4, 2}, 1.5f);

    const auto view = image.hostView();

    static_assert(std::is_same_v<std::remove_cv_t<decltype(view)>, ImageView<const float>>);
    static_assert(std::is_same_v<decltype(view.pixels), const float *>);
    EXPECT_EQ(view.pixels, image.pixels.hostPtr());
    EXPECT_EQ(view.size.width, 4);
    EXPECT_EQ(view.size.height, 2);
    EXPECT_EQ(view.pixels[5], 1.5f);
}

TEST(ImageTest, MutableViewsConvertToReadOnlyViewsButNotTheOtherWayAround)
{
    static_assert(std::is_convertible_v<ImageView<float>, ImageView<const float>>);
    static_assert(!std::is_convertible_v<ImageView<const float>, ImageView<float>>);
    static_assert(!std::is_convertible_v<ImageView<float>, ImageView<const Vec4>>);

    auto image = Image1f::create(Size2i{2, 3}, 3.0f);

    const ImageView<const float> read_only = image.hostView();

    EXPECT_EQ(read_only.pixels, image.pixels.hostPtr());
    EXPECT_EQ(read_only.size.width, 2);
    EXPECT_EQ(read_only.size.height, 3);
}

TEST(ImageTest, IsMoveOnlyAndMovingKeepsTheSamePixelBuffer)
{
    static_assert(!std::is_copy_constructible_v<Image4f>);
    static_assert(std::is_move_constructible_v<Image4f>);

    auto source = Image1f::create(Size2i{2, 2}, 1.5f);
    const float *buffer = source.pixels.hostPtr();

    const Image1f moved = std::move(source);

    EXPECT_EQ(moved.pixels.hostPtr(), buffer);
    EXPECT_EQ(moved.size.width, 2);
    EXPECT_EQ(moved.size.height, 2);
    EXPECT_EQ(pixelsOf(moved), std::vector<float>(4, 1.5f));
}

// --- intensity ---

TEST(IntensityTest, EachPixelBecomesItsLuminanceAndSizeIsKept)
{
    auto image = Image4f::create(Size2i{2, 1});
    image[pixelAt(0, 0)] = Vec4{1, 0, 0, 1};
    image[pixelAt(1, 0)] = Vec4{0, 1, 0, 1};

    const Image1f result = intensity(image);

    EXPECT_EQ(result.size.width, 2);
    EXPECT_EQ(result.size.height, 1);
    EXPECT_FLOAT_EQ(result[pixelAt(0, 0)], luminance(Vec4{1, 0, 0, 1}));
    EXPECT_FLOAT_EQ(result[pixelAt(1, 0)], luminance(Vec4{0, 1, 0, 1}));
}

// --- savePNG / loadHDR ---

TEST(LoadHdrTest, MissingFileThrowsAndNamesTheFile)
{
    const std::string missing_path = "test_output/test_image/does_not_exist.png";

    try
    {
        loadHDR(missing_path);
        FAIL() << "Expected loadHDR to throw";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_NE(std::string(error.what()).find(missing_path), std::string::npos) << error.what();
    }
}

TEST(LoadHdrTest, SavedPngRoundTripsFullyOnAndFullyOffChannels)
{
    const auto opaque_red = Vec4{1, 0, 0, 1};
    const auto opaque_blue = Vec4{0, 0, 1, 1};
    const auto path = savedPngWithOnePixelPerColor("round_trip.png", {opaque_red, opaque_blue});

    const Image4f loaded = loadHDR(path);

    ASSERT_EQ(loaded.size.width, 2);
    ASSERT_EQ(loaded.size.height, 1);
    test::expectVec4Near(loaded[pixelAt(0, 0)], opaque_red);
    test::expectVec4Near(loaded[pixelAt(1, 0)], opaque_blue);
}

TEST(LoadHdrTest, LoadsTheCheckedInReferenceRender)
{
    const Image4f reference = loadHDR("test_input/triangle_umbrella.png");

    EXPECT_EQ(reference.size.width, 256);
    EXPECT_EQ(reference.size.height, 256);
    EXPECT_EQ(reference.pixels.size(), static_cast<size_t>(reference.size.area()));
}

// --- writeImageMetadata ---

TEST(ImageMetadataTest, RenderStatisticsCanBeReadBackFromTheImage)
{
    const auto path = savedPngWithOnePixelPerColor("with_metadata.png", {Vec4{0, 0, 0, 1}});
    const ImageMetadata metadata{
        .totalRayCasts = 12'345'678'901ull,
        .raysPerPixel = 2.5,
        .maxPathDepth = 10,
        .pixelSampling = "UniformRandom",
        .outputLinearity = "GammaCorrected",
    };

    ASSERT_TRUE(writeImageMetadata(path, metadata));

    EXPECT_EQ(xmpValue(path, "Xmp.devilray.TotalRayCasts"), "12345678901");
    EXPECT_EQ(std::stod(xmpValue(path, "Xmp.devilray.RaysPerPixel").value()), 2.5);
    EXPECT_EQ(xmpValue(path, "Xmp.devilray.MaxPathDepth"), "10");
    EXPECT_EQ(xmpValue(path, "Xmp.devilray.PixelSampling"), "UniformRandom");
    EXPECT_EQ(xmpValue(path, "Xmp.devilray.OutputLinearity"), "GammaCorrected");
}

TEST(ImageMetadataTest, EmptyTextFieldsAreOmittedRatherThanWrittenBlank)
{
    const auto path = savedPngWithOnePixelPerColor("without_text_fields.png", {Vec4{0, 0, 0, 1}});

    ASSERT_TRUE(writeImageMetadata(path, ImageMetadata{}));

    EXPECT_FALSE(xmpValue(path, "Xmp.devilray.PixelSampling").has_value());
    EXPECT_FALSE(xmpValue(path, "Xmp.devilray.OutputLinearity").has_value());
    EXPECT_TRUE(xmpValue(path, "Xmp.devilray.TotalRayCasts").has_value());
}

TEST(ImageMetadataTest, ImageIsCreditedThroughStandardExifTags)
{
    const auto path = savedPngWithOnePixelPerColor("with_exif.png", {Vec4{0, 0, 0, 1}});

    ASSERT_TRUE(writeImageMetadata(path, ImageMetadata{}));

    EXPECT_TRUE(hasExifTag(path, "Exif.Image.Artist"));
    EXPECT_TRUE(hasExifTag(path, "Exif.Image.Software"));
}

TEST(ImageMetadataTest, ReturnsFalseWhenTheImageCannotBeOpened)
{
    const auto missing_path = (outputFolder() / "does_not_exist.png").string();

    EXPECT_FALSE(writeImageMetadata(missing_path, ImageMetadata{}));
}

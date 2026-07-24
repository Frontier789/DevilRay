#include "ImageMetadata.hpp"

#include <exiv2/exiv2.hpp>

#include <iostream>

#if __has_include(<GitVersion.hpp>)
#  include <GitVersion.hpp>
#endif

namespace
{
    constexpr const char *kXmpNamespace = "https://github.com/Frontier789/DevilRay/ns/1.0/";
    constexpr const char *kXmpPrefix = "devilray";
    constexpr const char *author = "Matyas Komaromi";
    constexpr const char *software = "DevilRay 0.1";
}

bool writeImageMetadata(const std::string &path, const ImageMetadata &meta)
{
    try
    {
        auto image = Exiv2::ImageFactory::open(path);
        image->readMetadata();

        // Standard Exif tags
        Exiv2::ExifData &exif = image->exifData();
        exif["Exif.Image.Artist"] = author;
        exif["Exif.Image.Software"] = software;

        // Render statistics under
        Exiv2::XmpProperties::registerNs(kXmpNamespace, kXmpPrefix);
        Exiv2::XmpData &xmp = image->xmpData();
        xmp["Xmp.devilray.TotalRayCasts"] = std::to_string(meta.totalRayCasts);
        xmp["Xmp.devilray.RaysPerPixel"] = std::to_string(meta.raysPerPixel);
        xmp["Xmp.devilray.MaxPathDepth"] = std::to_string(meta.maxPathDepth);
        if (!meta.pixelSampling.empty()) xmp["Xmp.devilray.PixelSampling"] = meta.pixelSampling;
        if (!meta.outputLinearity.empty()) xmp["Xmp.devilray.OutputLinearity"] = meta.outputLinearity;

#ifdef DEVILRAY_GIT_COMMIT
        xmp["Xmp.devilray.GitCommit"] = DEVILRAY_GIT_COMMIT;
        xmp["Xmp.devilray.GitBranch"] = DEVILRAY_GIT_BRANCH;
#endif
#ifdef DEVILRAY_GIT_ORIGIN
        xmp["Xmp.devilray.ReposLink"] = DEVILRAY_GIT_ORIGIN;
#endif

        image->writeMetadata();
        return true;
    }
    catch (const Exiv2::Error &e)
    {
        std::cout << "WARN: Failed to write metadata to " << path << ": " << e.what() << std::endl;
        return false;
    }
}

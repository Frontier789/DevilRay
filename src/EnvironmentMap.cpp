#include "tracing/EnvironmentMap.hpp"

EnvironmentMapView EnvironmentMap::view()
{
    return EnvironmentMapView{
        .intensity_multiplier = intensity_multiplier,
        .panorama_image = panorama_image.view()
    };
}

#pragma once

#include "Image.hpp"

struct EnvironmentMapView;

struct EnvironmentMap
{
    float intensity_multiplier = 1.0f;

    Image4f panorama_image;

    EnvironmentMapView view();
};

struct EnvironmentMapView
{
    float intensity_multiplier = 1.0f;

    ImageView4f panorama_image;
};



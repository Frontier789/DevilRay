#pragma once

enum class OutputLinearity
{
    Linear,
    GammaCorrected
};

struct OutputOptions
{
    OutputLinearity linearity;
};
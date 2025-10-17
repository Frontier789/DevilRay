#pragma once

struct Medium
{
    float ior; ///< Index of refraction

    static const Medium air;
};

inline const Medium Medium::air{.ior = 1.0003f};

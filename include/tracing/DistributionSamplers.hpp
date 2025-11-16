#pragma once

#include "Utils.hpp"

#pragma nv_exec_check_disable
template<typename Rng>
HD Vec3 uniformHemisphereSample(const Vec3 &normal, Rng &r)
{
    const float theta0 = 2 * pi * r.rnd();
    const float theta1 = std::acos(1 - 2 * r.rnd());

    const float x = std::sin(theta1) * std::sin(theta0);
    const float y = std::sin(theta1) * std::cos(theta0);
    const float z = std::cos(theta1);

    const auto v = Vec3{x,y,z};

    if (dot(v, normal) < 0) return v*-1;

    return v;
}

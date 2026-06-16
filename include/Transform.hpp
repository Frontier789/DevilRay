#pragma once

#include <Utils.hpp>

struct Transform
{
    Vec3 s = Vec3{1,1,1};
    Vec3 p = Vec3{0,0,0};

    constexpr Ray applyInverse(const Ray &ray_in_world) const;
    constexpr Vec3 applyToPoint(const Vec3 &point) const;
};

constexpr Ray Transform::applyInverse(const Ray &ray_in_world) const
{
    const auto inv_s = this->s.inv();
    return Ray{
        .p = (ray_in_world.p - this->p) * inv_s,
        .v = ray_in_world.v * inv_s,
    };
}

constexpr Vec3 Transform::applyToPoint(const Vec3 &point) const
{
    return point * this->s + this->p;
}

#pragma once

#include "device/Array.hpp"
#include "Utils.hpp"

#include <span>

#pragma nv_exec_check_disable
template<typename Rng>
HD Vec3 uniformSphereSample(Rng &r)
{
    const float theta0 = 2 * pi * r.rnd();
    const float theta1 = std::acos(1 - 2 * r.rnd());

    const float x = std::sin(theta1) * std::sin(theta0);
    const float y = std::sin(theta1) * std::cos(theta0);
    const float z = std::cos(theta1);

    return Vec3{x,y,z};
}

#pragma nv_exec_check_disable
template<typename Rng>
HD Vec3 uniformHemisphereSample(const Vec3 &normal, Rng &r)
{
    const auto v = uniformSphereSample(r);

    if (v.dot(normal) < 0) return v*-1;

    return v;
}

#pragma nv_exec_check_disable
template<typename Rng>
HD Vec3 cosineWeightedHemisphereSample(const Vec3 &normal, Rng &r)
{
    const auto v = uniformSphereSample(r);

    const auto direction = v + normal;

    return direction.normalized();
}

struct AliasEntry
{
    float p_A;
    float pdf_A;
    float pdf_B;
    int A;
    int B;
};

struct AliasTable
{
    DeviceArray<AliasEntry> entries;
};

struct AliasSample
{
    int index;
    float pdf;
};

template<typename Rng>
HD AliasSample sample(std::span<const AliasEntry> table, Rng &rng)
{
    const float r = rng.rnd();
    float findex = r * table.size();
    if (findex == table.size()) findex = table.size()-1;

    const int index = static_cast<int>(findex);

    const float p = rng.rnd();

    const auto &entry = table[index];

    if (p <= entry.p_A) {
        return AliasSample{
            .index = entry.A,
            .pdf = entry.pdf_A
        };
    }

    return AliasSample{
        .index = entry.B,
        .pdf = entry.pdf_B
    };
}

AliasTable generateAliasTable(std::span<const float> importances);

template<typename Rng>
HD Vec3 uniformTriangleSample(const Vec3 &A, const Vec3 &B, const Vec3 &C, Rng &rng)
{
    float u = rng.rnd();
    float v = rng.rnd();

    if (u+v > 1) {
        u = 1-u;
        v = 1-v;
    }

    return A + (B-A) * u + (C-A)*v;
}
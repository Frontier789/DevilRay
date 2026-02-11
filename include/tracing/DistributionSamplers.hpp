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

#pragma nv_exec_check_disable
template<typename Rng>
HD Vec3 uniformHemisphereSample(const Vec3 &normal, Rng &r)
{
    const auto v = uniformSphereSample(r);

    if (dot(v, normal) < 0) return v*-1;

    return v;
}

#pragma nv_exec_check_disable
template<typename Rng>
HD Vec3 cosineWeightedHemisphereSample(const Vec3 &normal, Rng &r)
{
    const auto v = uniformSphereSample(r);

    return (v + normal).normalized();
}

struct AliasEntry
{
    float p_A;
    int A;
    int B;
};

struct AliasTable
{
    DeviceArray<AliasEntry> entries;
};

template<typename Rng>
HD int sample(std::span<const AliasEntry> table, Rng &rng)
{
    const float r = rng.rnd();
    const float findex = r * table.size();
    const int index = static_cast<int>(findex);
    
    const float p = rng.rnd();

    const auto &entry = table[index];

    // printf("Rng gave index %d\n\tWhich has p_A=%f\n\tP=%f\n\tA=%d B=%d\n", index, entry.p_A, p, entry.A, entry.B);

    if (p <= entry.p_A) {
        return entry.A;
    }

    return entry.B;
}

AliasTable generateAliasTable(std::span<const float> importances);

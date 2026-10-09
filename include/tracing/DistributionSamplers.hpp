#pragma once

#include "device/Array.hpp"
#include "Utils.hpp"
#include "Image.hpp"

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

struct AliasTableView
{
    const AliasEntry *entries;

    int entry_count;
};

struct AliasTable
{
    DeviceArray<AliasEntry> entries;

    AliasTableView view()
    {
        entries.ensureDeviceAllocation();

        return AliasTableView{
            .entries = entries.devicePtr(),
            .entry_count = static_cast<int>(entries.size()),
        };
    }

    AliasTableView hostView() const
    {
        return AliasTableView{
            .entries = entries.hostPtr(),
            .entry_count = static_cast<int>(entries.size()),
        };
    }
};

struct AliasSample
{
    int index;
    float pdf;
};

template<typename Rng>
HD AliasSample sample(const AliasTableView &table, Rng &rng)
{
    const float r = rng.rnd();
    float findex = r * table.entry_count;
    if (findex == table.entry_count) findex = table.entry_count-1;

    const int index = static_cast<int>(findex);

    const float p = rng.rnd();

    const auto &entry = table.entries[index];

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

struct AliasImageTableView
{
    const AliasEntry *pixel_entries;
    const AliasEntry *row_entries;

    Size2i image_size;
};

struct AliasImageTable
{
    DeviceArray<AliasEntry> pixel_entries;
    DeviceArray<AliasEntry> row_entries;

    Size2i image_size;

    AliasImageTableView view()
    {
        pixel_entries.ensureDeviceAllocation();
        row_entries.ensureDeviceAllocation();

        return AliasImageTableView{
            .pixel_entries = pixel_entries.devicePtr(),
            .row_entries = row_entries.devicePtr(),
            .image_size = image_size,
        };
    }

    AliasImageTableView hostView() const
    {
        return AliasImageTableView{
            .pixel_entries = pixel_entries.hostPtr(),
            .row_entries = row_entries.hostPtr(),
            .image_size = image_size,
        };
    }
};

struct AliasImageSample
{
    Vec2i pixel_coordinate;
    float pdf;
};

template<typename Rng>
HD AliasImageSample sample(const AliasImageTableView &table, Rng &rng)
{
    const auto s = table.image_size;

    const auto row_table = AliasTableView{
        .entries = table.row_entries,
        .entry_count = s.height,
    };
    const auto row = sample(row_table, rng);

    const auto col_table = AliasTableView{
        .entries = table.pixel_entries + s.width * row.index,
        .entry_count = s.width,
    };
    const auto col = sample(col_table, rng);

    return AliasImageSample{
        .pixel_coordinate = Vec2i{.x = col.index, .y = row.index},
        .pdf = row.pdf * col.pdf,
    };
}

AliasImageTable generateAliasTable(const ImageView1f &image);

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
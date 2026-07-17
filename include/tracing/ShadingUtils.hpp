#pragma once

#include "device/DevUtils.hpp"

struct MisPdfs {
    float bsdf_pdf;
    float nee_pdf;
};

inline HD float computeNeePdf(
    const Vec3 &prevPos,
    const Vec3 &pos,
    const Vec3 &normal,
    const Material &material,
    const ObjectsInfo &info)
{
    const auto next_radiant_exitance = luminance(radiantExitance(material));
    const auto pdf_nee_area = next_radiant_exitance / info.total_radiant_power;
    const float nee_pdf = pdf_nee_area * areaToSolidAngle(prevPos, pos, normal);

    return nee_pdf;
}

inline HD Vec4 misWeightedEmission(
    const Vec4 &emission,
    const Vec4 &throughput,
    bool prev_bounce_specular,
    MisPdfs path_pdfs)
{
    if (prev_bounce_specular) {
        return emission * throughput;
    } else if (path_pdfs.bsdf_pdf + path_pdfs.nee_pdf > 0) {
        return emission * throughput * powerHeuristic(path_pdfs.bsdf_pdf, path_pdfs.nee_pdf);
    }
    return Vec4{0,0,0,0};
}

struct LightSample
{
    Vec3 p;
    Vec3 n;
    int mat;
    float pdf;
};

#pragma nv_exec_check_disable
template<typename Rng>
HD LightSample samplePointOnLights(
    std::span<const TriangleMesh> objects,
    std::span<const AliasEntry> light_table,
    Rng &rng)
{
    const auto [index, object_pdf] = sample(light_table, rng);
    const auto &object = objects[index];
    const auto mat = object.material;

    // printf("Rolled %d\n", index);

    const auto tris_table = std::span{object.triangle_sampler, static_cast<size_t>(object.triangle_count)};
    const auto [i, triangle_pdf] = sample(tris_table, rng);

    const auto triangle = object.triangles[i];
    const auto A = object.points[triangle.a.pi];
    const auto B = object.points[triangle.b.pi];
    const auto C = object.points[triangle.c.pi];

    const auto p = uniformTriangleSample(A, B, C, rng);

    const auto Aw = A * object.model_to_world.s;
    const auto Bw = B * object.model_to_world.s;
    const auto Cw = C * object.model_to_world.s;

    const auto perp = (Aw - Bw).cross(Aw - Cw);
    const auto perp_length = perp.length();
    const auto n = perp / perp_length;
    const auto world_triangle_area = perp_length / 2.0f;

    return LightSample{
        .p = object.model_to_world.applyToPoint(p),
        .n = n,
        .mat = mat,
        .pdf = object_pdf * triangle_pdf / world_triangle_area,
    };
}

inline HD Vec4 evaluateDirectLighting(
    const Vec3 &surface_pos,
    const Vec3 &surface_normal,
    const Vec4 &diffuse_reflectance,
    const LightSample &light_sample,
    const Vec4 &light_emission,
    std::span<const TriangleMesh> objects
)
{
    if (occludedScene(surface_pos, light_sample.p, objects))
        return Vec4{0,0,0,0};

    const auto distance = (light_sample.p - surface_pos).length();
    const auto w_light = (light_sample.p - surface_pos) / distance;

    const auto brdf = diffuse_reflectance / pi;

    const auto cos_at_surface = w_light.dot(surface_normal);
    const auto cos_at_light   = w_light.dot(light_sample.n);

    if (cos_at_surface <= 0)
        return Vec4{0,0,0,0};

    const auto geometric_term = cos_at_surface * std::abs(cos_at_light) / (distance * distance);
    return light_emission * brdf * geometric_term / light_sample.pdf;
}

#include "tracing/TriangleMesh.hpp"

void TriangleMesh::setPosition(const Vec3 &pos)
{
    this->p = pos;
}

void TriangleMesh::setScale(const Vec3 &scale)
{
    this->s = scale;

    const float vol = scale.x * scale.y * scale.z;
    const float area_factor = std::cbrt(vol * vol);
    this->surface_area = this->base_surface_area * area_factor;
}

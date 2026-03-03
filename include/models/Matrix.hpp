#pragma once

#include "Utils.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

template<typename T>
struct Matrix4x4
{
    T values[4][4];

    constexpr Matrix4x4 operator*(const Matrix4x4 &mat) const;
    static constexpr Matrix4x4 translation(const Vec3 &offset);
    static constexpr Matrix4x4 rotation(const Vec3 &axis, const float angleInRadians);

    constexpr Vec3 applyToPosition(const Vec3 &pos) const;
    constexpr Vec3 applyToDirection(const Vec3 &dir) const;

    constexpr Vec3 getOffset() const;

    static constexpr Matrix4x4<T> identity();
};

template<typename T>
inline std::ostream &operator<<(std::ostream &out, const Matrix4x4<T> &m)
{
    const auto &v = m.values;

    size_t w = 0;

    for (const auto &r : v) for (const auto &x : r) {
        std::stringstream ss;
        ss << x;

        w = std::max(w, ss.str().length()+1);
    }

    out << std::setw(w) << v[0][0] << "," <<  std::setw(w) << v[0][1] << "," <<  std::setw(w) << v[0][2] << "," <<  std::setw(w) << v[0][3] << "\n";
    out << std::setw(w) << v[1][0] << "," <<  std::setw(w) << v[1][1] << "," <<  std::setw(w) << v[1][2] << "," <<  std::setw(w) << v[1][3] << "\n";
    out << std::setw(w) << v[2][0] << "," <<  std::setw(w) << v[2][1] << "," <<  std::setw(w) << v[2][2] << "," <<  std::setw(w) << v[2][3] << "\n";
    out << std::setw(w) << v[3][0] << "," <<  std::setw(w) << v[3][1] << "," <<  std::setw(w) << v[3][2] << "," <<  std::setw(w) << v[3][3];

    return out;
}

typedef Matrix4x4<float> Matrix4x4f;

template<typename T>
constexpr Matrix4x4<T> Matrix4x4<T>::identity()
{
    return Matrix4x4<T>{
        {{1,0,0,0},
         {0,1,0,0},
         {0,0,1,0},
         {0,0,0,1}}
    };
}

template<typename T>
constexpr Matrix4x4<T> Matrix4x4<T>::operator*(const Matrix4x4<T> &mat) const {
    Matrix4x4<T> result{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            result.values[r][c] = values[r][0] * mat.values[0][c] +
                                  values[r][1] * mat.values[1][c] +
                                  values[r][2] * mat.values[2][c] +
                                  values[r][3] * mat.values[3][c];
        }
    }
    return result;
}

template<typename T>
constexpr Matrix4x4<T> Matrix4x4<T>::translation(const Vec3 &offset) {
    return Matrix4x4<T>{
        .values = {
            {1, 0, 0, offset.x},
            {0, 1, 0, offset.y},
            {0, 0, 1, offset.z},
            {0, 0, 0, 1}
    }};
}

template<typename T>
constexpr Vec3 Matrix4x4<T>::getOffset() const {
    return Vec3{
        .x = values[0][3],
        .y = values[1][3],
        .z = values[2][3],
    };
}

template<typename T>
constexpr Matrix4x4<T> Matrix4x4<T>::rotation(const Vec3 &axis, const float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const float t = 1.0f - c;
    const Vec3 a = axis;

    return Matrix4x4<T>{
        .values = {
            {t * a.x * a.x + c,       t * a.x * a.y - s * a.z, t * a.x * a.z + s * a.y, 0},
            {t * a.x * a.y + s * a.z, t * a.y * a.y + c,       t * a.y * a.z - s * a.x, 0},
            {t * a.x * a.z - s * a.y, t * a.y * a.z + s * a.x, t * a.z * a.z + c,       0},
            {0,                       0,                       0,                       1},
    }};
}

template<typename T>
constexpr Vec3 Matrix4x4<T>::applyToPosition(const Vec3 &pos) const {
    return Vec3{
        .x = values[0][0] * pos.x + values[0][1] * pos.y + values[0][2] * pos.z + values[0][3],
        .y = values[1][0] * pos.x + values[1][1] * pos.y + values[1][2] * pos.z + values[1][3],
        .z = values[2][0] * pos.x + values[2][1] * pos.y + values[2][2] * pos.z + values[2][3],
    };
}

template<typename T>
constexpr Vec3 Matrix4x4<T>::applyToDirection(const Vec3 &dir) const {
    return Vec3{
        .x = values[0][0] * dir.x + values[0][1] * dir.y + values[0][2] * dir.z,
        .y = values[1][0] * dir.x + values[1][1] * dir.y + values[1][2] * dir.z,
        .z = values[2][0] * dir.x + values[2][1] * dir.y + values[2][2] * dir.z,
    };
}

#pragma once

#include "tracing/Camera.hpp"

struct CameraController
{
    Camera camera;

    Vec3 target;
    float pitch;
    float yaw;
    float distance;

    Matrix4x4f calculateTransform() const;

    Camera getCamera() const;

    void handleDrag(Vec2f offset_in_pixels);
    void handleRotate(Vec2f offset_in_pixels);
    void handleScroll(float amount);

    Vec3 position() const;
    Vec3 forward() const;
    Vec3 up() const;
    Vec3 right() const;
};
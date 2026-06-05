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
    Matrix4x4f getViewMatrix() const;

    Camera getCamera() const;
    bool isUpsideDown() const;

    void handleDrag(Vec2f offset_in_pixels);
    void handleRotate(Vec2f offset_in_pixels, Vec2f sensitivity = Vec2f{0.005f, 0.007f});
    void handleScroll(float amount);

    Vec3 position() const;
    Vec3 forward() const;
    Vec3 up() const;
    Vec3 right() const;
};
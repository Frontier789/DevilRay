#include "CameraController.hpp"

Vec3 CameraController::forward() const {
    return Vec3{
        std::cos(pitch) * std::sin(yaw),
        std::sin(pitch),
        std::cos(pitch) * std::cos(yaw)
    };
}

Vec3 CameraController::right() const {
    Vec3 f = forward();
    Vec3 worldUp = {0.0f, 1.0f, 0.0f};
    
    // Cross product: worldUp x f
    return Vec3{
        worldUp.y * f.z - worldUp.z * f.y,
        worldUp.z * f.x - worldUp.x * f.z,
        worldUp.x * f.y - worldUp.y * f.x
    }.normalized(); // Ensure the result is a unit vector
}

Vec3 CameraController::up() const {
    const auto f = forward();
    const auto r = right();
    
    return f.cross(r);
}

Vec3 CameraController::position() const {
    return target - forward() * distance;
}

Matrix4x4f CameraController::calculateTransform() const {
    const auto f = forward();
    const auto r = right();
    const auto u = up();
    const auto p = position();

    return Matrix4x4f{
        .values = {
            {r.x, u.x, f.x, p.x},
            {r.y, u.y, f.y, p.y},
            {r.z, u.z, f.z, p.z},
            {  0,   0,   0,   1},
        }
    };
}

void CameraController::handleRotate(Vec2f offset_in_pixels)
{
    pitch += offset_in_pixels.y * -0.007f;
    yaw += offset_in_pixels.x * 0.005f;
}

void CameraController::handleDrag(Vec2f offset_in_pixels)
{
    const auto u = up();

    const auto d = distance;
    const auto f = camera.intrinsics.focal_length;
    const auto px = camera.physical_pixel_size.toVec();

    const auto offset_on_sensor = offset_in_pixels * px;
    const auto offset_on_target_plane = offset_on_sensor * (d / f);
    
    const auto right = forward().cross(u);

    const auto offset_in_space = u * offset_on_target_plane.y + right * offset_on_target_plane.x;

    target += offset_in_space;
}

void CameraController::handleScroll(float amount)
{
    distance *= std::pow(0.8f, amount);
}
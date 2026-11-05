#pragma once

#include "CameraController.hpp"
#include "DebugOptions.hpp"
#include "Renderer.hpp"
#include "Utils.hpp"
#include "tracing/Camera.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/Scene.hpp"
#include "tracing/TriangleMesh.hpp"

#include <GL/glew.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <atomic>
#include <cfloat>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>

struct Meshes
{
    TriangleMesh suzanne;
    TriangleMesh cube;
    TriangleMesh lightPanel;
};

struct OGLObjects
{
    GLuint shader;
    GLuint texture;
    GLuint vao;
};

struct RenderOptions
{
    float focal_length_mm = 14.2f;
    DebugOptions debug = DebugOptions::Off;
    PixelSampling pixel_sampling = PixelSampling::UniformRandom;
    float environment_map_intensity_scale = 1.0f;
};

struct UiHandler
{
    ImVec2 currentMouse = ImVec2(-FLT_MAX,-FLT_MAX);
    bool mouseDown = false;
};

struct AsyncData
{
    std::atomic<bool> renderingNeedsReset;
    std::atomic<bool> renderingShouldStop;
    std::atomic<float> averageRenderTime;
    std::atomic<uint64_t> totalRayCasts;
};

struct Application
{
    static constexpr Size2i resolution{640, 640};
    static constexpr int render_scale = 1;
    static constexpr float physical_pixel_size = 3.72e-6 * 4 * render_scale;

    GLFWwindow *window;
    std::unique_ptr<Renderer> renderer;

    std::optional<Meshes> meshes;

    RenderOptions renderOptions;
    CameraController cameraController;
    UiHandler uiHandler;
    OGLObjects glObjects;

    std::jthread renderingThread;
    AsyncData asyncData;

    Application();
    ~Application();

    void handleUiEvents();
    void presentCurrentImage();

    void resetCamera();

private:
    void createWindow();
    void loadMeshes();
    void createOpenGLObjects();
    void initCameraController();
    void initUiHandler();
    void initRenderer();

    void handleCameraControl();

    void renderWorker();
};

Scene createScene(Meshes &meshes);

Camera createCamera(
    Size2i resolution,
    Vec3 position,
    const float focal_length_mm,
    const float physical_pixel_size
);
#include "Utils.hpp"
#include "tracing/GpuTris.hpp"
#include "DebugOptions.hpp"
#include "tracing/PixelSampling.hpp"
#include "Renderer.hpp"
#include "RunningAverage.hpp"
#include "CameraController.hpp"

#include <imgui_impl_glfw.h>
#include <GL/glew.h>

#include <thread>

struct Meshes
{
    GpuTris suzanne;
    GpuTris cube;
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
};

struct UiHandler
{
    ImVec2 currentMouse = ImVec2(-FLT_MAX,-FLT_MAX);
    bool mouseDown = false;
};

struct Application
{
    static constexpr Size2i resolution{640, 640};
    static constexpr int render_scale = 1;
    static constexpr float physical_pixel_size = 3.72e-6 * 4 * render_scale;

    GLFWwindow *window;
    std::unique_ptr<Renderer> renderer;

    Meshes meshes;

    RunningAverage renderTimes;
    RenderOptions renderOptions;
    CameraController cameraController;
    UiHandler uiHandler;
    OGLObjects glObjects;

    std::jthread renderingThread;
    std::atomic<bool> renderingNeedsReset;
    std::atomic<bool> renderingShouldStop;
    std::atomic<float> averageRenderTime;

    Application();
    ~Application();

    void handleUiEvents();
    void presentCurrentImage();

private:
    void createWindow();
    void loadMeshes();
    void createOpenGLObjects();
    void initCameraController();
    void initUiHandler();
    void initRenderer();

    void renderWorker();
};

Scene createScene(Meshes &meshes);

Camera createCamera(
    Size2i resolution,
    Vec3 position,
    const float focal_length_mm,
    const float physical_pixel_size
);
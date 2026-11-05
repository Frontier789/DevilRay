#pragma once

#include "CameraController.hpp"
#include "Utils.hpp"
#include "models/BBH.hpp"
#include "models/Matrix.hpp"
#include "models/Mesh.hpp"
#include "tracing/Camera.hpp"

#include <GL/glew.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <cfloat>
#include <optional>

struct OGLObjects
{
    GLuint meshShader;
    GLuint meshVbo;
    GLuint meshVao;
    GLsizei meshTrisBegin;
    GLsizei meshTrisEnd;

    GLuint bboxShader;
    GLuint bboxVbo;
    GLuint bboxVao;
    GLsizei bboxVertexCount = 0;
};

struct UiHandler
{
    ImVec2 currentMouse = ImVec2(-FLT_MAX,-FLT_MAX);
    bool mouseDown = false;
};

struct Application
{
    static constexpr Size2i resolution{1024, 768};

    static constexpr float focalLengthMm     = 50.0f;
    static constexpr float physicalPixelSize = 100e-6f;
    static constexpr float initialDistance   = 2.0f;

    static constexpr float nearPlane = 0.1f;
    static constexpr float farPlane  = 100.0f;

    GLFWwindow *window;

    Mesh mesh;
    std::optional<BBH> bbh;

    CameraController cameraController;
    UiHandler uiHandler;
    OGLObjects glObjects;

    int bbhShowDepth = 0;
    int boxCountOnDepth = 1;
    int boxShown = 0;
    bool showBbh = true;

    Application();
    ~Application();

    void drawUiElements();
    void handleUiEvents();
    void renderCurrentFrame();

private:
    void createWindow();
    void loadMesh();
    void uploadMeshToGpu();
    void createOpenGLObjects();
    void initCameraController();
    void initUiHandler();
    void updateBoundingBoxMesh();

    void handleCameraControl();

    void updateBbhDepth(int newDepth);
    void updateTrisShownBox();

    static Matrix4x4f perspectiveMatrix(float fovDeg, float aspect, float near, float far);
};

Camera createCamera(
    Size2i resolution,
    Vec3 position,
    const float focal_length_mm,
    const float physical_pixel_size
);

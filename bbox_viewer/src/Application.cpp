#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstddef>

#include "Application.hpp"
#include "Utils.hpp"
#include "Shaders.hpp"
#include "models/BBH.hpp"

void glfwErrorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

void GLAPIENTRY
oglErrorCallback(GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION ||
        type == GL_DEBUG_TYPE_PERFORMANCE)
    {
        return;
    }

    fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
           ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
            type, severity, message );
}

void initOGLDebug()
{
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(oglErrorCallback, 0);
}

void setGLVersion()
{
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 16);
}

void initGLFW()
{
    std::cout << "TRACE: initGLFW" << std::endl;

    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        std::exit(1);
    }

    setGLVersion();
}

void initGLEW(GLFWwindow *window)
{
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW!" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        std::exit(1);
    }
}

void initIMGUI(GLFWwindow *window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void Application::createWindow()
{
    std::cout << "TRACE: createWindow" << std::endl;

    window = glfwCreateWindow(resolution.width, resolution.height, "DevilRay", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        std::exit(1);
    }

    glfwMakeContextCurrent(window);

    initGLEW(window);
    initIMGUI(window);
    initOGLDebug();

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
}

void Application::loadMesh()
{
    std::cout << "TRACE: loadMesh" << std::endl;

    const std::string mesh_file = "models/bunny.obj";

    this->mesh = ::loadMesh(mesh_file);

    generateCoarseNormals(this->mesh);
    std::cout << "Mesh '" << mesh.name << "' has " << mesh.points.size() << " points" << std::endl;
    std::cout << "Mesh '" << mesh.name << "' has " << mesh.normals.size() << " normals" << std::endl;
    std::cout << "Mesh '" << mesh.name << "' has " << mesh.triangles.size() << " tris" << std::endl;

    normalizeMeshSize(this->mesh);

    this->bbh = generateSimpleBBH(this->mesh);
    updateTrisShownBox();
}

namespace {
    struct GPUVertex {
        Vec3 position;
        Vec3 normal;
    };
}

void Application::uploadMeshToGpu()
{
    std::vector<GPUVertex> vertices;
    vertices.reserve(mesh.triangles.size() * 3);
    for (const auto &tri : mesh.triangles)
    {
        for (const Vertex &v : {tri.a, tri.b, tri.c})
            vertices.push_back({mesh.points[v.pi], mesh.normals[v.ni]});
    }

    glNamedBufferData(
        glObjects.meshVbo,
        vertices.size() * sizeof(GPUVertex),
        vertices.data(),
        GL_STATIC_DRAW
    );
}

static void appendBboxLineVerts(const AABB &bbox, std::vector<Vec3> &vertices)
{
    const Vec3 &lo = bbox.min;
    const Vec3 &hi = bbox.max;

    const Vec3 corners[8] = {
        {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z},
        {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z},
        {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z},
        {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z},
    };

    const Vec3 edges[24] = {
        // bottom face
        corners[0], corners[1],  corners[1], corners[2],
        corners[2], corners[3],  corners[3], corners[0],
        // top face
        corners[4], corners[5],  corners[5], corners[6],
        corners[6], corners[7],  corners[7], corners[4],
        // vertical edges
        corners[0], corners[4],  corners[1], corners[5],
        corners[2], corners[6],  corners[3], corners[7],
    };

    vertices.insert(vertices.end(), std::begin(edges), std::end(edges));
}

void Application::updateBoundingBoxMesh()
{
    std::vector<Vec3> lineVerts;

    for (const auto &node : getBoxesOnDepth(*bbh, bbhShowDepth))
        appendBboxLineVerts(node.box, lineVerts);

    glObjects.bboxVertexCount = static_cast<GLsizei>(lineVerts.size());

    glNamedBufferData(glObjects.bboxVbo, lineVerts.size() * sizeof(Vec3), lineVerts.data(), GL_STATIC_DRAW);
}

void Application::createOpenGLObjects()
{
    std::cout << "TRACE: createOpenGLObjects" << std::endl;

    glObjects.meshShader = createShaderProgram(passthroughVertexShader, passthroughFragmentShader);
    glCreateBuffers(1, &glObjects.meshVbo);
    glGenVertexArrays(1, &glObjects.meshVao);

    glBindVertexArray(glObjects.meshVao);
    glBindBuffer(GL_ARRAY_BUFFER, glObjects.meshVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GPUVertex), (void*)offsetof(GPUVertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GPUVertex), (void*)offsetof(GPUVertex, normal));
    glBindVertexArray(0);

    glObjects.bboxShader = createShaderProgram(solidColorVertexShader, solidColorFragmentShader);
    glCreateBuffers(1, &glObjects.bboxVbo);
    glGenVertexArrays(1, &glObjects.bboxVao);

    glBindVertexArray(glObjects.bboxVao);
    glBindBuffer(GL_ARRAY_BUFFER, glObjects.bboxVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glBindVertexArray(0);
}

Matrix4x4f Application::perspectiveMatrix(float fovDeg, float aspect, float near, float far)
{
    const float f = 1.0f / std::tan(fovDeg * 3.14159265358979f / 180.0f / 2.0f);
    const float A = (far + near) / (near - far);
    const float B = 2.0f * far * near / (near - far);

    return Matrix4x4f{.values = {
        {f / aspect, 0.0f, 0.0f,  0.0f},
        {0.0f,       f,    0.0f,  0.0f},
        {0.0f,       0.0f, A,     B   },
        {0.0f,       0.0f, -1.0f, 0.0f},
    }};
}

void Application::renderCurrentFrame()
{
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    const float aspect = float(resolution.width) / float(resolution.height);

    // Derive vertical FOV from the physical camera intrinsics
    const float focalLengthM    = Application::focalLengthMm / 1000.0f;
    const float sensorHalfHeight = resolution.height / 2.0f * Application::physicalPixelSize;
    const float fovDeg = 2.0f * std::atan(sensorHalfHeight / focalLengthM) * 180.0f / 3.14159265358979f;

    const Matrix4x4f proj = perspectiveMatrix(fovDeg, aspect, Application::nearPlane, Application::farPlane);

    const Matrix4x4f mvp = proj * cameraController.getViewMatrix();

    glLineWidth(2);

if (showBbh) {
    glUseProgram(glObjects.bboxShader);
    glUniformMatrix4fv(0, 1, GL_TRUE, &mvp.values[0][0]);
    glBindVertexArray(glObjects.bboxVao);

    if (glObjects.bboxVertexCount > 0)
    {
        glDepthMask(GL_FALSE);

        glUniform4f(1, 0.3f, 0.3f, 0.3f, 1.0f);
        glDrawArrays(GL_LINES, 0, glObjects.bboxVertexCount);

        glDepthMask(GL_TRUE);
    }
}

{
    glUseProgram(glObjects.meshShader);
    glUniformMatrix4fv(0, 1, GL_TRUE, &mvp.values[0][0]);
    glBindVertexArray(glObjects.meshVao);
    const auto numberOfTrianglesToDraw = glObjects.meshTrisEnd - glObjects.meshTrisBegin;
    glDrawArrays(GL_TRIANGLES, glObjects.meshTrisBegin * 3, numberOfTrianglesToDraw * 3);
    glBindVertexArray(0);
}

if (showBbh) {
    glUseProgram(glObjects.bboxShader);
    glUniformMatrix4fv(0, 1, GL_TRUE, &mvp.values[0][0]);
    glBindVertexArray(glObjects.bboxVao);

    if (glObjects.bboxVertexCount > 0)
    {
        glUniform4f(1, 1.0f, 0.75f, 0.2f, 1.0f);
        if (boxShown < 0) {
            glDrawArrays(GL_LINES, 0, glObjects.bboxVertexCount);
        } else {
            const auto vertexPerBox = 12 * 2;
            glDrawArrays(GL_LINES, boxShown * vertexPerBox, vertexPerBox);
        }
    }
}

    glBindVertexArray(0);
}

Camera createCamera(
    Size2i resolution,
    Vec3 position,
    const float focal_length_mm,
    const float physical_pixel_size
){
    return Camera{
        .transform = Matrix4x4f::translation(position * -1),
        .intrinsics = Intrinsics{
            .focal_length = focal_length_mm / 1000.0f,
            .center = Vec2f{
                resolution.width  / 2.0f * physical_pixel_size,
                resolution.height / 2.0f * physical_pixel_size,
            },
        },
        .resolution = resolution,
        .physical_pixel_size = Size2f{physical_pixel_size, physical_pixel_size},
    };
}

void Application::initCameraController()
{
    std::cout << "TRACE: initCameraController" << std::endl;

    constexpr float initialDistance = Application::initialDistance;

    cameraController = CameraController{
        .camera = createCamera(resolution, Vec3{0, 0, -initialDistance}, Application::focalLengthMm, Application::physicalPixelSize),
        .target = Vec3{0, 0, 0},
        .pitch = 0,
        .yaw = 0,
        .distance = initialDistance,
    };
}

void Application::initUiHandler()
{
    std::cout << "TRACE: initUiHandler" << std::endl;

    uiHandler.currentMouse = ImGui::GetMousePos();
    uiHandler.mouseDown = false;
}

Application::Application()
{
    initGLFW();

    createWindow();
    createOpenGLObjects();
    initCameraController();
    initUiHandler();
    loadMesh();
    uploadMeshToGpu();
    updateBoundingBoxMesh();
}

Application::~Application()
{
    glDeleteProgram(glObjects.meshShader);
    glDeleteVertexArrays(1, &glObjects.meshVao);
    glDeleteBuffers(1, &glObjects.meshVbo);

    glDeleteProgram(glObjects.bboxShader);
    glDeleteVertexArrays(1, &glObjects.bboxVao);
    glDeleteBuffers(1, &glObjects.bboxVbo);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

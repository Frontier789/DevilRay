#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>

#include "Application.hpp"
#include "Utils.hpp"
#include "Shaders.hpp"

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
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
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

GpuTris loadMeshAndPrint(const std::string &file)
{
    auto mesh = loadMesh(file);
    
    std::cout << "Mesh '" << mesh.name << "' has " << mesh.points.size() << " points" << std::endl;
    std::cout << "Mesh '" << mesh.name << "' has " << mesh.normals.size() << " normals" << std::endl;
    std::cout << "Mesh '" << mesh.name << "' has " << mesh.triangles.size() << " tris" << std::endl;
    
    return convertMeshToTris(mesh);
}

void Application::loadMeshes()
{
    std::cout << "TRACE: loadMeshes" << std::endl;

    meshes.suzanne = loadMeshAndPrint("models/suzanne.obj");
    meshes.cube = loadMeshAndPrint("models/cube.obj");
}

GLuint createTexture(Size2i resolution)
{
    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);

    glTextureStorage2D(texture, 1, GL_RGBA8, resolution.width, resolution.height);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    return texture;
}

void Application::createOpenGLObjects()
{
    std::cout << "TRACE: createOpenGLObjects" << std::endl;

    glObjects.shader = createShaderProgram(fullScreenQuadVertexShader, fullScreenQuadFragmentShader);
    glObjects.texture = createTexture(resolution / render_scale);

    glGenVertexArrays(1, &glObjects.vao);
}

Camera createCamera(
    Size2i resolution,
    Vec3 position,
    const float focal_length_mm,
    const float physical_pixel_size
){
    Camera cam{
        .transform = Matrix4x4f::translation(position * -1),
        .intrinsics = Intrinsics{
            .focal_length = focal_length_mm / 1000.0f,
            .center = Vec2{
                        resolution.width/2.f * physical_pixel_size,
                        resolution.height/2.f * physical_pixel_size,
            }
        },
        .resolution = resolution,
        .physical_pixel_size = Size2f{physical_pixel_size, physical_pixel_size},
    };

    return cam;
}

void Application::initCameraController()
{
    std::cout << "TRACE: initCameraController" << std::endl;

    cameraController = CameraController{
        .camera = createCamera(resolution / render_scale, Vec3{}, renderOptions.focal_length_mm, physical_pixel_size),
        .target = Vec3{0,0,2},
        .pitch = 0,
        .yaw = 0,
        .distance = 2,
    };
}

void Application::initUiHandler()
{
    std::cout << "TRACE: initUiHandler" << std::endl;

    uiHandler.currentMouse = ImGui::GetMousePos();
    uiHandler.mouseDown = false;
}

void Application::initRenderer()
{
    std::cout << "TRACE: initRenderer" << std::endl;

    renderer = std::make_unique<Renderer>(resolution / render_scale);

    renderer->setScene(createScene(meshes));
    cameraController.camera.transform = cameraController.calculateTransform(); // TODO: automate this
    renderer->setCamera(cameraController.camera);
    renderer->setDebug(renderOptions.debug);

    renderingThread = std::jthread([this]{
        renderWorker();
    });
}

Application::Application()
    : renderTimes(20)
{
    initGLFW();

    createWindow();
    createOpenGLObjects();
    initCameraController();
    initUiHandler();
    loadMeshes();
    initRenderer();
}

Application::~Application()
{
    asyncData.renderingShouldStop.store(true, std::memory_order::relaxed);
    renderingThread.join();

    glDeleteProgram(glObjects.shader);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void Application::presentCurrentImage()
{
    glTextureSubImage2D(glObjects.texture, 0, 0, 0, resolution.width / render_scale, resolution.height / render_scale, GL_RGBA, GL_UNSIGNED_BYTE, renderer->getPixels());

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(glObjects.vao);
    glUseProgram(glObjects.shader);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, glObjects.texture);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

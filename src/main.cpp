#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>

#include "Application.hpp"

template<typename DrawCallback>
void runEventLoop(Application &app, DrawCallback drawCallback)
{
    using namespace std::chrono;
    auto lastFpsPrint = steady_clock::now();
    int framesSinceLastPrint = 0;
    std::string fpsString = "? fps";

    while (!glfwWindowShouldClose(app.window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ImGui demo window (for testing purposes)
        // ImGui::ShowDemoWindow();

        ImGui::Text("%s", fpsString.c_str());

        drawCallback();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(app.window);

        ++framesSinceLastPrint;
        auto t = steady_clock::now();
        if (duration_cast<milliseconds>(t - lastFpsPrint) > 1s)
        {
            fpsString = std::to_string(framesSinceLastPrint) + " fps";
            framesSinceLastPrint = 0;
            lastFpsPrint += 1s;
        }
    }
}

const char* vertexShaderSource = R"glsl(
#version 330 core

void main() {
    const vec2 vertices[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0)
    );
    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core

out vec4 frag_color;

void main() {
    frag_color = vec4(0.0, 0.0, 1.0, 1.0);
}
)glsl";

GLuint compileShader(const char *source, GLenum type)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLsizei infoLogLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        std::vector<char> infoLogArr(infoLogLength);
        glGetShaderInfoLog(shader, infoLogLength, nullptr, infoLogArr.data());
        std::string infoLog = infoLogArr.data();
        
        std::cerr << "Failed to compile shader:\n" << infoLog << std::endl;

        throw std::runtime_error("Failed to compile shader");
    }

    return shader;
}

GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLsizei infoLogLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

        std::vector<char> infoLogArr(infoLogLength);
        glGetProgramInfoLog(program, infoLogLength, nullptr, infoLogArr.data());
        std::string infoLog = infoLogArr.data();
        
        std::cerr << "Failed to link shader:\n" << infoLog << std::endl;

        throw std::runtime_error("Failed to link shader");
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}


int main() {
    auto app = initApplication();
    
    try
    {
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
        const auto shader = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    
        GLuint vao;
        glGenVertexArrays(1, &vao);
        
        runEventLoop(app, [&]{
            
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            
            glBindVertexArray(vao);
            glUseProgram(shader);
            glDrawArrays(GL_TRIANGLES, 0, 3);
    
        });
    
        glDeleteProgram(shader);
    }
    catch (std::exception &e)
    {
        std::cout << "Got exception, closing window: " << e.what() << std::endl;
        closeApplication(app);
        
        std::throw_with_nested(e);
    }
}
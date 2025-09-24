#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>

#include "Application.hpp"
#include "Shaders.hpp"
#include <cmath>

void onKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_Q)
    {
        glfwSetWindowShouldClose(window, true);
    }
}

template<typename DrawCallback>
void runEventLoop(Application &app, DrawCallback drawCallback)
{
    using namespace std::chrono;
    auto lastFpsPrint = steady_clock::now();
    int framesSinceLastPrint = 0;
    std::string fpsString = "? fps";

    glfwSetKeyCallback(app.window, onKeyEvent);

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
#version 430 core

out vec2 tex_coord;

void main() {
    const vec2 vertices[4] = vec2[4](
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0)
    );

    tex_coord = vertices[gl_VertexID]/2.0 + 0.5;
    
    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 430 core

in vec2 tex_coord;

layout(location = 0)
uniform sampler2D tex;

out vec4 frag_color;

void main() {
    frag_color = texture(tex, tex_coord);
}
)glsl";

GLuint createTexture()
{
    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);

    glTextureStorage2D(texture, 1, GL_RGBA8, 800, 600);

    return texture;
}

void setTextureData(GLuint texture)
{
    std::vector<uint32_t> pixels(800*600);

    auto packPixel = [](float r, float g, float b, float a) -> uint32_t {
        return (static_cast<uint32_t>(std::clamp(r, 0.0f, 1.0f) * 255)<< 0) | 
               (static_cast<uint32_t>(std::clamp(g, 0.0f, 1.0f) * 255)<< 8) | 
               (static_cast<uint32_t>(std::clamp(b, 0.0f, 1.0f) * 255)<<16) | 
               (static_cast<uint32_t>(std::clamp(a, 0.0f, 1.0f) * 255)<<24);
    };

    auto setPixel = [&](int x, int y, float r, float g, float b, float a=1.0f) {
        pixels[x + y*800] = packPixel(r,g,b,a);
    };

    for (int y=0;y<600;++y)
    for (int x=0;x<800;++x)
    {
        float intensity = std::sin(y / 30.0f * 2 * std::numbers::pi_v<float>)/2 + 0.5;
    
        setPixel(x, y, intensity, intensity, intensity);
    }

    glTextureSubImage2D(texture, 0, 0, 0, 800, 600, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
}

int main() {
    auto app = initApplication();
    
    try
    {
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
        const auto shader = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    
        GLuint vao;
        glGenVertexArrays(1, &vao);

        const auto texture = createTexture();
        setTextureData(texture);
        
        runEventLoop(app, [&]{
            
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            
            glBindVertexArray(vao);
            glUseProgram(shader);
            
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
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
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <filesystem>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <deque>

#include "Application.hpp"
#include "Shaders.hpp"
#include "Image.hpp"
#include "Utils.hpp"
#include "Renderer.hpp"
#include "CameraController.hpp"
#include "tracing/Material.hpp"
#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/LightSampling.hpp"
#include "tracing/GpuTris.hpp"
#include "models/Mesh.hpp"

/*
    PLAN
     - bidirectional path tracing
        - need to be able to sample light sources
     - importance sampling
        - report and use pdf
     - metropolis
     - keep track of variance
        - add option to show it
     - stratify sample pixels
*/

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

        if (ImGui::IsKeyPressed(ImGuiKey_Q) && !ImGui::GetIO().WantTextInput)
            glfwSetWindowShouldClose(app.window, true);

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


int main() {
    printCudaDeviceInfo();

    Application app;

    try {
        runEventLoop(app, [&]{
            app.handleUiEvents();
            
            app.presentCurrentImage();
        });
    }
    catch (const std::exception &e)
    {
        std::cout << "Got exception: " << e.what() << std::endl;

        return 1;
    }
}

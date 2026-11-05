#include "Application.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>

template<typename DrawCallback>
void runEventLoop(Application &app, DrawCallback drawCallback)
{
    using namespace std::chrono;
    auto lastFpsPrint = steady_clock::now();
    int framesSinceLastPrint = 0;
    std::string fpsString = "? fps";

    glfwSwapInterval(1);

    while (!glfwWindowShouldClose(app.window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");

        if (ImGui::IsKeyPressed(ImGuiKey_Q) && !ImGui::GetIO().WantTextInput)
            glfwSetWindowShouldClose(app.window, true);


        ImGui::Text("%s", fpsString.c_str());

        drawCallback();

        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(app.window);

        ++framesSinceLastPrint;
        auto t = steady_clock::now();
        if (duration_cast<milliseconds>(t - lastFpsPrint) > 1s)
        {
            fpsString = std::to_string(framesSinceLastPrint) + " fps";
            framesSinceLastPrint = 0;
            lastFpsPrint = t;
        }
    }
}


int main() {
    Application app;

    try {
        runEventLoop(app, [&]{
            app.drawUiElements();
            app.handleUiEvents();

            app.renderCurrentFrame();
        });
    }
    catch (const std::exception &e)
    {
        std::cout << "Got exception: " << e.what() << std::endl;

        return 1;
    }
}

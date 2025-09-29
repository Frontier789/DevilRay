#include <imgui_impl_glfw.h>

#include "Utils.hpp"

struct Application
{
    GLFWwindow *window;
};

Application initApplication(Size2i resolution);

void closeApplication(Application &app);
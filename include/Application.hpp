#include <imgui_impl_glfw.h>

struct Application
{
    GLFWwindow *window;
};

Application initApplication();

void closeApplication(Application &app);
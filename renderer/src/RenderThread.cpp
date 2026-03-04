#include "Application.hpp"

void Application::renderWorker();
{
    while (!renderingShouldStop.load(std::memory_order::relaxed))
    {
        renderer->render();
    }
}
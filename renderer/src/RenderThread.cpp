#include "Application.hpp"

void Application::renderWorker()
{
    while (!asyncData.renderingShouldStop.load(std::memory_order::relaxed))
    {
        renderer->render();

        asyncData.averageRenderTime.store(renderer->getMeanRenderTimes());
    }
}
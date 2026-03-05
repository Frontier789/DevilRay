#include "Application.hpp"

void Application::renderWorker()
{
    while (!asyncData.renderingShouldStop.load(std::memory_order::relaxed))
    {
        Timer t;
        renderer->render();
        const auto elapsed_ms = t.elapsed_seconds() * 1000;
        renderTimes.add(elapsed_ms);

        asyncData.averageRenderTime.store(renderTimes.mean(), std::memory_order::relaxed);
    }
}
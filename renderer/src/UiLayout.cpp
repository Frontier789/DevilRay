#include "Application.hpp"

std::string counterToString(uint64_t cntr)
{
    const auto original = cntr;

    for (const auto ending : {"", "K", "M", "G", "T"}) {
        if (cntr < 1000) return std::to_string(cntr) + ending;

        cntr /= 1000;
    }

    return std::to_string(cntr);
}

void Application::handleUiEvents()
{
    if (ImGui::SliderFloat("Focal length", &renderOptions.focal_length_mm, 3, 150, "%.1f mm"))
    {
        cameraController.camera = createCamera(resolution / render_scale, Vec3{}, renderOptions.focal_length_mm, physical_pixel_size);

        renderer->setCamera(cameraController.getCamera());
    }

    constexpr const char* debug_names[] = {"Off", "UVChecker", "BariCoords", "WindingOrder"};
    if (ImGui::Combo("Debug", reinterpret_cast<int*>(&renderOptions.debug), debug_names, IM_ARRAYSIZE(debug_names)))
    {
        renderer->setDebug(renderOptions.debug);
    }

    constexpr const char* pixel_sampling_names[] = {"Center", "UniformRandom"};
    if (ImGui::Combo("Pixel Sampling", reinterpret_cast<int*>(&renderOptions.pixel_sampling), pixel_sampling_names, IM_ARRAYSIZE(pixel_sampling_names)))
    {
        renderer->setPixelSampling(renderOptions.pixel_sampling);
    }

    ImGui::Text("Render pass: %.1fms", asyncData.averageRenderTime.load(std::memory_order::relaxed));

    // {
    //     std::scoped_lock guard{renderingMutex};

    //     auto &buffers = renderer->getBuffers();
    //     buffers.casts.updateHostData();
    
    //     ImGui::Text("Rays per pixel: %s", counterToString(buffers.totalCasts() / resolution.width / resolution.height).c_str());
    // }
    
    if (ImGui::Button("Capture snapshot"))
    {
        const auto imageFolder = std::filesystem::path{"captures"};
        (void)std::filesystem::create_directory(imageFolder);

        std::time_t time = std::time({});
        char timeString[100];
        std::strftime(timeString, 100, "%Y_%m_%d_%H_%M.png", std::gmtime(&time));
        std::string timePng = timeString;

        std::cout << "Saving to " << imageFolder / timePng << std::endl;

        renderer->saveImage(imageFolder / timePng);
    }

    const auto mouse = ImGui::GetMousePos();
    if (ImGui::IsMousePosValid(&mouse))
    {
        const auto io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            const auto wheel = io.MouseWheel;
            if (wheel != 0) {
                cameraController.handleScroll(wheel);
    
                renderer->setCamera(cameraController.getCamera());
            }
    
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
            {
                if (!uiHandler.mouseDown) {
                    uiHandler.mouseDown = true;
                    uiHandler.currentMouse = mouse;
                }
    
                if (mouse.x != uiHandler.currentMouse.x || mouse.y != uiHandler.currentMouse.y) {
                    const auto dx = mouse.x - uiHandler.currentMouse.x;
                    const auto dy = mouse.y - uiHandler.currentMouse.y;
                    uiHandler.currentMouse = mouse;
    
                    const auto pixelOffset = Vec2f{dx, dy};
                    
                    if (io.KeyShift) {
                        cameraController.handleDrag(pixelOffset / render_scale);
                    } else {
                        cameraController.handleRotate(pixelOffset);
                    }

                    renderer->setCamera(cameraController.getCamera());
                }
            }
            else
            {
                uiHandler.mouseDown = false;
            }
        }
    }
}
#include "Application.hpp"

void Application::handleUiEvents()
{
    handleCameraControl();
}

void Application::handleCameraControl()
{
    const auto mouse = ImGui::GetMousePos();
    const auto io = ImGui::GetIO();
    
    if (!ImGui::IsMousePosValid(&mouse)) return;
    if (io.WantCaptureMouse) return;

    const auto wheel = io.MouseWheel;
    if (wheel != 0) {
        cameraController.handleScroll(wheel);
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
                cameraController.handleDrag(pixelOffset);
            } else {
                cameraController.handleRotate(pixelOffset);
            }
        }
    }
    else
    {
        uiHandler.mouseDown = false;
    }
}

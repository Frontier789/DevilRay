#include "Application.hpp"

void Application::drawUiElements()
{
    const float btnW = ImGui::GetFrameHeight();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    if (ImGui::Button("-##bbh", ImVec2(btnW, 0)))
    {
        addBbhDepth(-1);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-(btnW + spacing));
    if (ImGui::SliderInt("##bbh_depth", &bbhShowDepth, 0, this->bbh.depth, "BBH depth: %d"))
        updateBoundingBoxMesh();
    ImGui::SameLine();
    if (ImGui::Button("+##bbh", ImVec2(btnW, 0)))
    {
        addBbhDepth(+1);
    }

    ImGui::Checkbox("Show parent bbox", &uiHandler.showParentBbox);
}

void Application::handleUiEvents()
{
    handleCameraControl();
}

void Application::addBbhDepth(int delta)
{
    int newDepth = bbhShowDepth + delta;
    if (newDepth < 0) newDepth = 0;
    if (newDepth > this->bbh.depth) newDepth = this->bbh.depth;

    if (bbhShowDepth != newDepth)
    {
        bbhShowDepth = newDepth;
        updateBoundingBoxMesh();
    }
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

    if (ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_KeypadAdd, false))
    {
        addBbhDepth(+1);
    }

    if (ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_KeypadSubtract, false))
    {
        addBbhDepth(-1);
    }
}

#include "Application.hpp"

namespace
{
    template<typename F>
    void sliderIntWithPlusMinus(int currentValue, int maxValue, const std::string &label, F f)
    {
        int newValue = currentValue;

        const float btnW = ImGui::GetFrameHeight();
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        if (ImGui::Button(("-##" + label).c_str(), ImVec2(btnW, 0)))
        {
            newValue -= 1;
            f(newValue);
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-(btnW + spacing));
        if (ImGui::SliderInt(("##" + label).c_str(), &newValue, 0, maxValue-1, (label + ": %d").c_str()))
        {
            f(newValue);
        }
        ImGui::SameLine();
        if (ImGui::Button(("+##" + label).c_str(), ImVec2(btnW, 0)))
        {
            newValue += 1;
            f(newValue);
        }
    }
}

void Application::drawUiElements()
{
    sliderIntWithPlusMinus(bbhShowDepth, this->bbh.depth, "BBH depth", [this](int newDepth){updateBbhDepth(newDepth);});
    sliderIntWithPlusMinus(boxShown, boxCountOnDepth, "Box to show", [this](int newBoxId){
        if (newBoxId < -1) newBoxId = -1;
        boxShown = newBoxId;
        updateTrisShownBox();
    });

    ImGui::Checkbox("Show bounding boxes", &showBbh);
}

void Application::handleUiEvents()
{
    handleCameraControl();
}

void Application::updateBbhDepth(int newDepth)
{
    if (newDepth < 0) newDepth = 0;
    if (newDepth > this->bbh.depth) newDepth = this->bbh.depth;

    if (bbhShowDepth == newDepth) return;

    bbhShowDepth = newDepth;
    boxCountOnDepth = getBoxesOnDepth(bbh, bbhShowDepth).size();

    if (boxShown >= boxCountOnDepth) {
        boxShown = boxCountOnDepth - 1;
    }

    updateBoundingBoxMesh();
    updateTrisShownBox();
}

void Application::updateTrisShownBox()
{
    if (boxShown >= boxCountOnDepth) boxShown = boxCountOnDepth - 1;
    if (boxShown < 0) {
        glObjects.meshTrisBegin = 0;
        glObjects.meshTrisEnd = mesh.triangles.size();
        return;
    }

    const auto nodesOnDepth = getBoxesOnDepth(bbh, bbhShowDepth);
    const auto box = nodesOnDepth[boxShown];

    glObjects.meshTrisBegin = box.tris_begin;
    glObjects.meshTrisEnd = box.tris_end;
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
        updateBbhDepth(bbhShowDepth+1);
    }

    if (ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_KeypadSubtract, false))
    {
        updateBbhDepth(bbhShowDepth-1);
    }
}

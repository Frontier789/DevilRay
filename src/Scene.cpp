
#include "tracing/Scene.hpp"

void Scene::deleteDeviceMemory()
{
    materials.deleteDeviceMemory();
    objects.deleteDeviceMemory();
}

void Scene::ensureDeviceAllocation()
{
    materials.ensureDeviceAllocation();
    objects.ensureDeviceAllocation();
}

#include "Application.hpp"

Scene createScene(Meshes &meshes)
{
    Scene scene;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> coin;

    const int  light3 = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{3,3,3},
            .diffuse_reflectance = Vec4{1.0,1.0,1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.3, 0.4, 0.0};
        scene.materials.push_back(material);
    }

    const int  light_mid = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{4,4,4},
            .diffuse_reflectance = Vec4{1.0,1.0,1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.3, 0.4, 0.0},
        scene.materials.push_back(material);
    }

    const int  light_bright = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{3000,3000,3000},
            .diffuse_reflectance = Vec4{1.0,1.0,1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.3, 0.4, 0.0},
        scene.materials.push_back(material);
    }

    const int  red = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.9, 0.3, 0.3, 0.0},
        };
        material.debug_color = Vec4{0.8, 0.2, 0.2, 0.0},
        scene.materials.push_back(material);
    }

    const int  green = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.3, 0.9, 0.3, 0.0},
        };
        material.debug_color = Vec4{0.2, 0.8, 0.2, 0.0},
        scene.materials.push_back(material);
    }

    const int  blue = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.3, 0.3, 0.9, 0.0},
        };
        material.debug_color = Vec4{0.2, 0.2, 0.8, 0.0},
        scene.materials.push_back(material);
    }

    const int  white = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.9, 0.9, 0.9, 0.0},
        };
        material.debug_color = Vec4{0.7, 0.7, 0.7, 0.0},
        scene.materials.push_back(material);
    }

    const int  glass = scene.materials.size();
    {
        auto material = TransparentMaterial{
            .inside_medium = Medium{.ior = 1.5595f},
        };
        material.debug_color = Vec4{0.9, 0.9, 0.9, 0.0},
        scene.materials.push_back(material);
    }

    const int  air = scene.materials.size();
    {
        auto material = TransparentMaterial{
            .inside_medium = Medium{.ior = 1.0f},
        };
        material.debug_color = Vec4{0.9, 0.9, 0.9, 0.0},
        scene.materials.push_back(material);
    }

    auto addQuad = [&](Vec3 center, Vec3 normal, Vec3 right, float size, int mat) {
        scene.mesh_storage.push_back(createQuadMesh(center, normal, right, size));
        auto view = viewGpuTris(scene.mesh_storage.back());
        view.material = mat;
        scene.objects.push_back(std::move(view));
    };

    addQuad(Vec3{0,0,2.5}, Vec3{0,0,-1}, Vec3{1,0,0}, 0.6, white);
    addQuad(Vec3{0.5,0,2}, Vec3{-1,0,0}, Vec3{0,1,0}, 0.6, red);
    addQuad(Vec3{-0.5,0,2}, Vec3{1,0,0}, Vec3{0,1,0}, 0.6, green);
    addQuad(Vec3{0,0.5,2}, Vec3{0,-1,0}, Vec3{0,0,1}, 0.6, white);
    addQuad(Vec3{0,-0.5,2}, Vec3{0,1,0}, Vec3{0,0,1}, 0.6, white);

    {
        auto lightPanel = viewGpuTris(meshes.lightPanel);
        lightPanel.material = light_mid;
        lightPanel.setPosition(Vec3{0, 0.499f, 2});
        scene.objects.push_back(std::move(lightPanel));
    }

    {
        auto mesh_object_cube = viewGpuTris(meshes.cube);
        mesh_object_cube.material = white;
        mesh_object_cube.setPosition(Vec3{0.0, -0.7, 2});
        mesh_object_cube.setScale(Vec3{0.2f,0.2f,0.2f});
        scene.objects.push_back(std::move(mesh_object_cube));
    }

    return scene;
}

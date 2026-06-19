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

    const int  light_low = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{20,20,20},
            .diffuse_reflectance = Vec4{1.0,1.0,1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.3, 0.4, 0.0};
        scene.materials.push_back(material);
    }

    const int  light_mid = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{40,40,40},
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

    const int  gray = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.2, 0.2, 0.2, 0.0},
        };
        material.debug_color = Vec4{0.4, 0.7, 0.7, 0.0},
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
            .inside_medium = Medium{.ior = 1.2},
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

    addQuad(Vec3{0,0,2.5}, Vec3{0,0,-1}, Vec3{1,0,0}, 1, white);
    addQuad(Vec3{0.5,0,2}, Vec3{-1,0,0}, Vec3{0,1,0}, 1, red);
    addQuad(Vec3{-0.5,0,2}, Vec3{1,0,0}, Vec3{0,1,0}, 1, green);
    addQuad(Vec3{0,0.5,2}, Vec3{0,-1,0}, Vec3{0,0,1}, 1, white);
    addQuad(Vec3{0,-0.5,2}, Vec3{0,1,0}, Vec3{0,0,1}, 1, white);

    {
        auto lightPanel = viewGpuTris(meshes.lightPanel);
        lightPanel.material = light_mid;
        lightPanel.setPosition(Vec3{0, 0.4999f, 2});
        scene.objects.push_back(std::move(lightPanel));
    }

    // addQuad(Vec3{0,0.5,2}, Vec3{0,-1,0}, Vec3{0,0,1}, 100, light_mid);

    {
        auto mesh_object_monkey = viewGpuTris(meshes.suzanne);
        mesh_object_monkey.material = glass;
        mesh_object_monkey.setPosition(Vec3{0.0, -0.32, 2});
        mesh_object_monkey.setScale(Vec3{0.35f,0.35f,0.35f});
        scene.objects.push_back(std::move(mesh_object_monkey));
    }

    // addQuad(Vec3{0,10,0}, Vec3{0,1,0}, Vec3{0,0,1}, 1000, light_mid);

    // {
    //     auto mesh_object_cube = viewGpuTris(meshes.cube);
    //     mesh_object_cube.material = blue;
    //     mesh_object_cube.setPosition(Vec3{0.0, 0.2f, 2});
    //     mesh_object_cube.setScale(Vec3{0.05f,0.05f,0.05f});
    //     scene.objects.push_back(std::move(mesh_object_cube));
    // }

    // {
    //     std::uniform_real_distribution<float> px(-0.4f, 0.4f);
    //     std::uniform_real_distribution<float> py(-0.45f, -0.3f);
    //     std::uniform_real_distribution<float> pz( 1.8f, 2.2f);

    //     for (int i = 0; i < 10; ++i) {
    //         auto cube = viewGpuTris(meshes.cube);
    //         cube.material = light_low;
    //         cube.setPosition(Vec3{px(rng), py(rng), pz(rng)});
    //         cube.setScale(Vec3{0.01f, 0.01f, 0.01f});
    //         scene.objects.push_back(std::move(cube));
    //     }
    // }

    // {
    //     auto mesh_object_cube = viewGpuTris(meshes.cube);
    //     mesh_object_cube.material = green;
    //     mesh_object_cube.setPosition(Vec3{0.0, 0.0, 2.0});
    //     mesh_object_cube.setScale(Vec3{0.1f,0.1f,0.1f});
    //     scene.objects.push_back(std::move(mesh_object_cube));
    // }

    // {
    //     auto mesh_object_cube = viewGpuTris(meshes.cube);
    //     mesh_object_cube.material = gray;
    //     mesh_object_cube.setPosition(Vec3{0.0, -0.5, 2.0});
    //     mesh_object_cube.setScale(Vec3{0.25f,0.25f,0.25f});
    //     scene.objects.push_back(std::move(mesh_object_cube));
    // }

    // addQuad(Vec3{0,-0.5,0}, Vec3{0,1,0}, Vec3{0,0,1}, 100, gray);

    return scene;
}

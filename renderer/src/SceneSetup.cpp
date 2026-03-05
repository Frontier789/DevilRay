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

    // Avoid GCC false positive warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstringop-overflow"
/*
    scene.objects.push_back(Sphere{
        .center = Vec3{0, 0, 2},
        .radius = 300e-3,
        .mat = blue,
    });

    scene.objects.push_back(Sphere{
        .center = Vec3{0.3, 0.2, 2},
        .radius = 100e-3,
        .mat = blue,
    });

    scene.objects.push_back(Sphere{
        .center = Vec3{-.3, .2, 2},
        .radius = 100e-3,
        .mat = blue,
    });
*/
    {
        auto obj = Square{
            .p = Vec3{0,0,2.5},
            .n = Vec3{0,0,-1},
            .right = Vec3{1,0,0},
            .size = 1,
        };
        obj.mat = white;
        scene.objects.push_back(std::move(obj));    
    }

    {
        auto obj = Square{
            .p = Vec3{0.5,0,2},
            .n = Vec3{1,0,0},
            .right = Vec3{0,1,0},
            .size = 1,
        };
        obj.mat = red;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Square{
            .p = Vec3{-0.5,0,2},
            .n = Vec3{1,0,0},
            .right = Vec3{0,1,0},
            .size = 1,
        };
        obj.mat = green;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Square{
            .p = Vec3{0,0.5,2},
            .n = Vec3{0,1,0},
            .right = Vec3{0,0,1},
            .size = 1,
        };
        obj.mat = white;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Square{
            .p = Vec3{0,-0.5,2},
            .n = Vec3{0,1,0},
            .right = Vec3{0,0,1},
            .size = 1,
        };
        obj.mat = white;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Square{
            .p = Vec3{0,0.499,2},
            .n = Vec3{0,1,0},
            .right = Vec3{0,0,1},
            .size = 0.5,
        };
        obj.mat = light_mid;
        scene.objects.push_back(std::move(obj));
    }

    // {
    //     const int N = 5;
    //     const float totalSize = 0.5;
    //     const float padding = 0.05;
    //     const float individualSize = (0.5 - (N-1)*padding) / N;
        
    //     for (int x=0;x<N;++x) {
    //         for (int y=0;y<N;++y) {
    //             auto obj = Square{
    //                 .p = Vec3{
    //                     totalSize * -0.5f + (individualSize + padding) * x + individualSize * 0.5f,
    //                     0.499f,
    //                     2 + totalSize * -0.5f + (individualSize + padding) * y + individualSize * 0.5f
    //                 },
    //                 .n = Vec3{0,1,0},
    //                 .right = Vec3{0,0,1},
    //                 .size = individualSize,
    //             };
    //             obj.mat = light_mid;
    //             scene.objects.push_back(std::move(obj));
    //         }
    //     }
    // }

    // {
    //     auto obj = Sphere{
    //         .center = Vec3{0.3, -0.4, 2},
    //         .radius = 100e-3,
    //     };
    //     obj.mat = blue;
    //     scene.objects.push_back(std::move(obj));
    // }

    // {
    //     auto obj = Sphere{
    //         .center = Vec3{0.47, -0.4, 2.47},
    //         .radius = 1e-3,
    //     };
    //     obj.mat = light_bright;
    //     scene.objects.push_back(std::move(obj));
    // }

    // for (int x=0;x<10;++x) {
    //     for (int y=0;y<10;++y) {
    //         {
    //             float h = std::sin(x*152.48548 + y*1867.8613385 + 5168.4185674);
    //             auto obj = Sphere{
    //                 .center = Vec3{(x - 9/2.f) / 10.0f *0.8f, 0.1f + h*0.1f, 1.6f + y / 10.0f *0.8f},
    //                 .radius = 40e-3,
    //             };
    //             obj.mat = glass;
    //             scene.objects.push_back(std::move(obj));
    //         }
    //     }
    // }

    // {
    //     auto obj = Sphere{
    //         .center = Vec3{-0.25, -0.3, 1.8},
    //         .radius = 200e-3,
    //     };
    //     obj.mat = glass;
    //     scene.objects.push_back(std::move(obj));
    // }

    // {
    //     auto mesh_object_suzanne = viewGpuTris(meshes.suzanne);
    //     mesh_object_suzanne.mat = blue;
    //     mesh_object_suzanne.setPosition(Vec3{0.0, -0.3, 2});
    //     mesh_object_suzanne.setScale(Vec3{0.15f,0.15f,0.15f});
    //     scene.objects.push_back(std::move(mesh_object_suzanne));
    // }
    
    {
        auto mesh_object_cube = viewGpuTris(meshes.cube);
        mesh_object_cube.mat = white;
        mesh_object_cube.setPosition(Vec3{0.0, -0.7, 2});
        mesh_object_cube.setScale(Vec3{0.2f,0.2f,0.2f});
        scene.objects.push_back(std::move(mesh_object_cube));
    }


    #pragma GCC diagnostic pop

    return scene;
}

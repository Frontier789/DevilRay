#pragma once

namespace benchmark
{
    struct HitTests
    {
        int triangle_tests = 0;
        int bbox_tests = 0;

        constexpr void registerTriangleTest() { ++triangle_tests; }
        constexpr void registerBBoxTest() { ++bbox_tests; }
    };

    struct Skip
    {
        constexpr void registerTriangleTest() { }
        constexpr void registerBBoxTest() { }
    };
}

template <typename T>
concept Benchmark = requires(T b) { 
    b.registerTriangleTest();  
    b.registerBBoxTest();
};
#pragma once

namespace benchmark
{
    struct HitTests
    {
        int64_t triangle_tests = 0;
        int64_t bbox_tests = 0;

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
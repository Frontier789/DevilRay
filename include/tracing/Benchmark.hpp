#pragma once

namespace benchmark
{
    struct HitTests
    {
        int64_t triangle_tests = 0;
        int64_t triangle_hits = 0;
        int64_t bbox_tests = 0;

        constexpr void registerTriangleTest() { ++triangle_tests; }
        constexpr void registerTriangleHit() { ++triangle_hits; }
        constexpr void registerBBoxTest() { ++bbox_tests; }
    };

    struct Skip
    {
        constexpr void registerTriangleTest() { }
        constexpr void registerTriangleHit() { }
        constexpr void registerBBoxTest() { }
    };
}

template <typename T>
concept Benchmark = requires(T b) {
    b.registerTriangleTest();
    b.registerBBoxTest();
};
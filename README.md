# DevilRay

## How to build

1. Create a venv
2. Install python dependencies
3. conan install . --build=missing
4. cmake --preset conan-release
5. cmake --build --preset conan-release
6. build/Release/devil_ray

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps

class MyCudaGuiProjectConan(ConanFile):
    name = "devil_ray"
    version = "0.1"
    author = "Matyas Komaromi <matyas@komaro.me>"
    description = "Render raytracing using cuda."
    settings = "os", "compiler", "build_type", "arch"
    requires = (
        "glfw/3.4",
        "imgui/1.92.2b",
        "glew/2.2.0",
        "stb/cci.20240531"
    )
    default_options = {
        "glfw/*:shared": False,
        "imgui/*:shared": False,
        "glew/*:shared": False
    }

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        tc.variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()


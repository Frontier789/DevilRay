#include <GL/glew.h>

#include <iostream>
#include <vector>

GLuint compileShader(const char *source, GLenum type)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLsizei infoLogLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        std::vector<char> infoLogArr(infoLogLength);
        glGetShaderInfoLog(shader, infoLogLength, nullptr, infoLogArr.data());
        std::string infoLog = infoLogArr.data();

        std::cerr << "Failed to compile shader:\n" << infoLog << std::endl;

        throw std::runtime_error("Failed to compile shader");
    }

    return shader;
}

GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource)
{
    GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLsizei infoLogLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

        std::vector<char> infoLogArr(infoLogLength);
        glGetProgramInfoLog(program, infoLogLength, nullptr, infoLogArr.data());
        std::string infoLog = infoLogArr.data();

        std::cerr << "Failed to link shader:\n" << infoLog << std::endl;

        throw std::runtime_error("Failed to link shader");
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

const char* passthroughVertexShader = R"glsl(
#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
out vec3 va_normal;

layout(location = 0) uniform mat4 uMVP;

void main() {
    va_normal = normal;
    gl_Position = uMVP * vec4(position, 1.0);
}
)glsl";

const char* passthroughFragmentShader = R"glsl(
#version 430 core
in vec3 va_normal;
out vec4 frag_color;

void main() {
    vec3 n = normalize(va_normal);
    float d0 = max(dot(n, normalize(vec3( 1.0, -1.0,  0.5))), 0.0);
    float d1 = max(dot(n, normalize(vec3(-1.0, -0.3,  1.0))), 0.0);
    float d2 = max(dot(n, normalize(vec3( 0.0,  1.0, -0.3))), 0.0);
    vec3 color = vec3(0.42, 0.38, 0.32) * d0 * 0.55
               + vec3(0.28, 0.32, 0.40) * d1 * 0.35
               + vec3(0.30, 0.28, 0.26) * d2 * 0.20
               + vec3(0.7);
    frag_color = vec4(color, 1.0);
}
)glsl";

const char* solidColorVertexShader = R"glsl(
#version 430 core
layout(location = 0) in vec3 position;

layout(location = 0) uniform mat4 uMVP;

void main() {
    gl_Position = uMVP * vec4(position, 1.0);
}
)glsl";

const char* solidColorFragmentShader = R"glsl(
#version 430 core
layout(location = 1) uniform vec4 uColor;
out vec4 frag_color;

void main() {
    frag_color = uColor;
}
)glsl";

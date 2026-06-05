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
out vec3 va_pos;

layout(location = 0) uniform mat4 uMVP;

void main() {
    va_pos = position;
    gl_Position = uMVP * vec4(position, 1.0);
}
)glsl";

const char* passthroughFragmentShader = R"glsl(
#version 430 core
in vec3 va_pos;
out vec4 frag_color;

void main() {
    frag_color = vec4(sin(va_pos.x * 10.0)/2+0.5, sin(va_pos.y * 7.1 + 13.7)/2+0.5, 0, 1);
}
)glsl";

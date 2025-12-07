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
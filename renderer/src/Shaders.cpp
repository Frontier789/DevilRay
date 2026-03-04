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

const char* fullScreenQuadVertexShader = R"glsl(
#version 430 core

out vec2 tex_coord;

void main() {
    const vec2 vertices[4] = vec2[4](
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0)
    );

    tex_coord = vertices[gl_VertexID]*vec2(1,-1)/2.0 + 0.5;

    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
}
)glsl";

const char* fullScreenQuadFragmentShader = R"glsl(
#version 430 core

in vec2 tex_coord;

layout(location = 0)
uniform sampler2D tex;

out vec4 frag_color;

void main() {
    frag_color = texture(tex, tex_coord);
}
)glsl";
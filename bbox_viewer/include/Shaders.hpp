#include <GL/glew.h>

GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource);

extern const char *passthroughVertexShader;
extern const char *passthroughFragmentShader;

extern const char *solidColorVertexShader;
extern const char *solidColorFragmentShader;
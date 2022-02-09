#include "Shader.h"

#include <fstream>
#include <sstream>

#include "Utils/Log.h"
#include "Utils/Error.h"

namespace Graphics
{
   static GLuint compileShader(GLenum type, const char* source);
   static const char* shaderTypeToString(GLenum type);
   static void checkLinkErrors(GLuint program);

   GLuint createGraphicsShader(const char* vertex_source, const char* fragment_source)
   {
      GLuint vs = compileShader(GL_VERTEX_SHADER, vertex_source);
      GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragment_source);

      GL_CALL(GLuint program = glCreateProgram());
      GL_CALL(glAttachShader(program, vs));
      GL_CALL(glAttachShader(program, fs));
      GL_CALL(glLinkProgram(program));
      checkLinkErrors(program);

      GL_CALL(glDeleteShader(vs));
      GL_CALL(glDeleteShader(fs));

      return program;
   }

   GLuint createComputeShader(const char* compute_source)
   {
      GLuint cs = compileShader(GL_COMPUTE_SHADER, compute_source);

      GL_CALL(GLuint program = glCreateProgram());
      GL_CALL(glAttachShader(program, cs));
      GL_CALL(glLinkProgram(program));
      checkLinkErrors(program);

      GL_CALL(glDeleteShader(cs));

      return program;
   }

   GLuint compileShader(GLenum type, const char* source)
   {
      GL_CALL(GLuint id = glCreateShader(type));
      GL_CALL(glShaderSource(id, 1, &source, NULL));
      GL_CALL(glCompileShader(id));

      GLint success;
      GL_CALL(glGetShaderiv(id, GL_COMPILE_STATUS, &success));
      if (!success)
      {
         GLint length;
         GL_CALL(glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length));
         GLchar* msg = reinterpret_cast<GLchar*>(alloca(length * sizeof(GLchar)));
         GL_CALL(glGetShaderInfoLog(id, length, &length, msg));
         ERROR("Shader (", shaderTypeToString(type), ") compilation error: '", msg, "'.");
      }

      return id;
   }

   const char* shaderTypeToString(GLenum type)
   {
      switch (type)
      {
         case GL_VERTEX_SHADER:
            return "VERTEX";
         case GL_FRAGMENT_SHADER:
            return "FRAGMENT";
         case GL_COMPUTE_SHADER:
            return "COMPUTE";
         default:
            return "UNKNOWN";
      }
   }

   void checkLinkErrors(GLuint program)
   {
      GLint success;
      GL_CALL(glGetProgramiv(program, GL_LINK_STATUS, &success));
      if (!success)
      {
         GLint length;
         GL_CALL(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length));
         GLchar* msg = reinterpret_cast<GLchar*>(alloca(length * sizeof(GLchar)));
         GL_CALL(glGetProgramInfoLog(program, length, &length, msg));
         ERROR("Program link error: '{}'.", msg);
      }
   }

} // namespace Graphics

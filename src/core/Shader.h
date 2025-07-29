#pragma once

#define GLAD_GLX 0
extern "C" {
#include <glad/glad.h>
}

#include <unordered_map>
#include <string>
#include <memory>
#include <stdexcept>

namespace MLGL {
    class ShaderManager;

    class Shader {
        friend ShaderManager;
    public:
        ~Shader();

        void bindBuffer(GLuint binding_point, const std::string& name, GLuint buffer);
        void setUniform(const std::string& name, float value);
        void setUniform(const std::string& name, int value);
        void setUniform(const std::string& name, bool value);
        void dispatch(GLuint x, GLuint y = 1, GLuint z = 1);

    private:
        explicit Shader(const std::string& filepath);
        GLuint get();
        GLuint m_Program;
    };

    class ShaderManager {
    public:
        static ShaderManager& getInstance();
        std::shared_ptr<Shader> getShader(const std::string& filepath);

    private:
        std::unordered_map<std::string, std::shared_ptr<Shader>> m_Shaders;

        ShaderManager() = default;
        ShaderManager(const ShaderManager&) = delete;
        ShaderManager& operator=(const ShaderManager&) = delete;
    };
}
#pragma once

#define GLAD_GLX 0
extern "C" {
#include <glad/glad.h>
}

#include <unordered_map>
#include <string>
#include <memory>
#include <stdexcept>

namespace NNGL {
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
        static ShaderManager& getInstance() {
            static ShaderManager instance;
            return instance;
        }

        // Loads or retrieves an existing shader
        std::shared_ptr<Shader> getShader(const std::string& filepath) {
            auto it = m_Shaders.find(filepath);
            if (it != m_Shaders.end())
                return it->second;  // Already loaded

            m_Shaders[filepath] = std::shared_ptr<Shader>(new Shader(filepath));
            return m_Shaders[filepath];
        }

    private:
        std::unordered_map<std::string, std::shared_ptr<Shader>> m_Shaders;

        // Make constructor private for singleton
        ShaderManager() = default;
        ShaderManager(const ShaderManager&) = delete;
        ShaderManager& operator=(const ShaderManager&) = delete;
    };
}
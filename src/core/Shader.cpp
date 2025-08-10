#include "Shader.h"
#include "Logger.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <stdexcept>

namespace MLGL {
	Shader::Shader(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filepath);

        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        std::string source = buffer.str();
        if (source.empty()) throw std::runtime_error("Failed to read shader file:" + filepath);

        GLuint m_Shader = glCreateShader(GL_COMPUTE_SHADER);
        const char* sourcePtr = source.c_str();
        glShaderSource(m_Shader, 1, &sourcePtr, nullptr);
        glCompileShader(m_Shader);

        GLint success;
        glGetShaderiv(m_Shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(m_Shader, 512, nullptr, infoLog);
            throw std::runtime_error("Shader compilation failed for file: " + filepath + "\n" + infoLog);
        }

        m_Program = glCreateProgram();
        glAttachShader(m_Program, m_Shader);
        glLinkProgram(m_Program);

        glGetProgramiv(m_Program, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(m_Program, 512, nullptr, infoLog);
            throw std::runtime_error("Shader program linking failed :" + filepath + " \n" + infoLog );
        }

        glDeleteShader(m_Shader);
	}

    void Shader::bindBuffer(GLuint binding_point, const std::string& name, GLuint buffer) {
        GLuint index = glGetProgramResourceIndex(m_Program, GL_SHADER_STORAGE_BLOCK, name.c_str());
        if (index == GL_INVALID_INDEX) {
            throw std::runtime_error("Buffer block not found: " + name);
        }
        if (buffer == 0) {
            throw std::runtime_error("Not initialized buffer: " + name);
        }
        glShaderStorageBlockBinding(m_Program, index, binding_point);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point, buffer);
        
        // Log buffer binding
        LOG_TRACE("[SHADER BIND] Bound buffer " + std::to_string(buffer) + 
            " to binding point " + std::to_string(binding_point) + 
            " with name '" + name + "'");
    }

    // Set float uniform by name
    void Shader::setUniform(const std::string& name, float value) {
        GLint loc = glGetUniformLocation(m_Program, name.c_str());
        if (loc == -1) {
            throw std::runtime_error("Uniform not found: " + name);
        }
        glUseProgram(m_Program);
        glUniform1f(loc, value);
    }

    void Shader::setUniform(const std::string& name, int value) {
        glUseProgram(m_Program);  // Make sure shader is active
        GLint location = glGetUniformLocation(m_Program, name.c_str());
        if (location == -1) {
            throw std::runtime_error("Uniform not found: " + name);
        }
        glUniform1i(location, value);
    }

    void Shader::setUniform(const std::string& name, bool value) {
        glUseProgram(m_Program);  // Make sure shader is active
        GLint location = glGetUniformLocation(m_Program, name.c_str());
        if (location == -1) {
            throw std::runtime_error("Uniform not found: " + name);
        }
        glUniform1i(location, value ? 1 : 0);
    }

    // Dispatch compute shader with given workgroup sizes
    void Shader::dispatch(GLuint x, GLuint y, GLuint z) {
        glUseProgram(m_Program);
        glDispatchCompute(x, y, z);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        //GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        //while (true) {
        //    GLenum result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 100000);
        //    if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED)
        //        break;
        //}
        //glDeleteSync(fence);
        // Log compute shader dispatch
        LOG_TRACE("[COMPUTE DISPATCH] Executed compute shader with workgroups: " +
            std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z));
    }

	Shader::~Shader() {
        if(m_Program) glDeleteProgram(m_Program);
	}

	GLuint Shader::get() {
        return m_Program;
	}

    ShaderManager& ShaderManager::getInstance() {
        static ShaderManager instance;
        return instance;
    }

    std::shared_ptr<Shader> ShaderManager::getShader(const std::string& filepath) {
        auto it = m_Shaders.find(filepath);
        if (it != m_Shaders.end())
            return it->second;

        m_Shaders[filepath] = std::shared_ptr<Shader>(new Shader(filepath));
        return m_Shaders[filepath];
    }
}
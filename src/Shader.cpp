#include "Shader.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

namespace NNGL {
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
        glShaderStorageBlockBinding(m_Program, index, binding_point);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point, buffer);
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
            // Uniform not found in shader - handle error or ignore
            return;
        }
        glUniform1i(location, value);
    }

    // Dispatch compute shader with given workgroup sizes
    void Shader::dispatch(GLuint x, GLuint y, GLuint z) {
        glUseProgram(m_Program);
        glDispatchCompute(x, y, z);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

	Shader::~Shader() {
        if(m_Program) glDeleteProgram(m_Program);
	}

	GLuint Shader::get() {
        return m_Program;
	}
}
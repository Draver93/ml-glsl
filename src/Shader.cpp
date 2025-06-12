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

	Shader::~Shader() {
        if(m_Program) glDeleteProgram(m_Program);
	}

	GLuint Shader::get() {
        return m_Program;
	}
}
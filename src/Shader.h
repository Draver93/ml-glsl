#pragma once

#define NOMINMAX
#define GLAD_GLX 0
extern "C" {
#include <glad/glad.h>
}

#include <string>

namespace NNGL {
	class Shader {
	public:
		Shader(const std::string& filepath);
		~Shader();
		GLuint get();
	private:
		GLuint m_Program;
	};
}
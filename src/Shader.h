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


		void bindBuffer(GLuint binding_point, const std::string& name, GLuint buffer);
		void setUniform(const std::string& name, float value);
		void setUniform(const std::string& name, int value);

		void dispatch(GLuint x, GLuint y = 1, GLuint z = 1);

	private:
		GLuint get();
		GLuint m_Program;
	};
}
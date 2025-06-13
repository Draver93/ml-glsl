#pragma once

#define NOMINMAX
#define GLAD_GLX 0
extern "C" {
#include <glad/glad.h>
}

#include <GLFW/glfw3.h>
#include "ActivationFunctions.h"
#include <glm/glm.hpp>


namespace NNGL {
	class NeuralNetwork;

	class Layer {
	private:
		Layer(int width, int height, int batchSize, ActivationFnType type);

	public:
		~Layer();

		glm::uvec2 getSize() { return { m_Width , m_Height }; }
		void printHeatmap();
		void displayLayer(const std::string& layer_name);

	private:
		GLuint m_WeightBuffer;
		GLuint m_BiasBuffer;
		GLuint m_ActivationBuffer;
		GLuint m_PreactivationBuffer;
		GLuint m_DeltaBuffer;

		int m_Width; // aka input size
		int m_Height; // output size

		ActivationFnType m_ActivationFnType;

		friend NeuralNetwork;
	};
}


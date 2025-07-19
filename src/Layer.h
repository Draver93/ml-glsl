#pragma once

#define GLAD_GLX 0
extern "C" {
#include <glad/glad.h>
}

#include <GLFW/glfw3.h>
#include "ActivationFunctions.h"
#include <glm/glm.hpp>
#include "Matrix.h"


namespace NNGL {
	class NeuralNetwork;

	class Layer {
	private:
		Layer(int width, int height, int batchSize, ActivationFnType type);

	public:
		~Layer();

		glm::uvec2 getSize() { return { m_Width , m_Height }; }
		void printHeatmap();
		void displayLayer(const std::string& layerName);

	public:
		GLuint m_WeightBuffer;
		GLuint m_BiasBuffer;
		std::shared_ptr<Matrix> m_ActivationMat;
		GLuint m_PreactivationBuffer;
		GLuint m_DeltaBuffer;

		GLuint m_ADAM_MBuffer;
		GLuint m_ADAM_VBuffer;

		int m_Width; // aka input size
		int m_Height; // output size

		ActivationFnType m_ActivationFnType;

		friend NeuralNetwork;
	};
}


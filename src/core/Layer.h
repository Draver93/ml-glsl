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

		Layer(const char* data);
		const char* save();
		size_t getSaveSize() const;

	public:
		~Layer() { }

		glm::uvec2 getSize() { return { m_Width , m_Height }; }
		void printHeatmap();

	private:
		std::shared_ptr<Matrix> 
			m_PreactivationMat,
			m_ActivationMat,
			m_WeightMat,
			m_DeltaMat,
			m_BiasMat;
		std::shared_ptr<Matrix> m_ADAM_M_Mat, m_ADAM_V_Mat;

		int m_Width; // aka input size
		int m_Height; // output size
		int m_BatchSize;

		ActivationFnType m_ActivationFnType;

		friend NeuralNetwork;
	};
}


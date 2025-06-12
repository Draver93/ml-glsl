#pragma once

#include "Layer.h"
#include "Shader.h"

#include <vector>


namespace NNGL {
	class NeuralNetwork {
	public:
		NeuralNetwork();
		~NeuralNetwork();
	public:
		void addLayer(std::unique_ptr<Layer> layer);
		void train(const std::vector<float>& input_batch, const std::vector<float>& target_batch, float learningRate = 0.01f);
		void run();
		void load();
		void save();

	private:
		void init();

		void bindTrainingData(const std::vector<float>& input_batch, const std::vector<float>& target_batch);
		void forwardPass();
		void targetLayerLossCalc();
		void hiddenLayersLossCalc();
		void weightsAndBiasesUpdate(float learningRate);

	private:
		GLuint m_InputBuffer = 0;
		GLuint m_TargetBuffer = 0;

		std::unique_ptr<Shader> 
			m_ForwardPassCompute,
			m_OutputDeltaCompute,
			m_HiddenDeltasCompute,
			m_WeightsCompute,
			m_BiasesCompute;
	public:
		std::vector<std::unique_ptr<NNGL::Layer>> m_Layers;
	};
}
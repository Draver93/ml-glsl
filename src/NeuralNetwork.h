#pragma once

#include "Layer.h"
#include "Shader.h"
#include "Matrix.h"

#include <vector>


namespace NNGL {
	class NeuralNetwork {
	public:
		NeuralNetwork(int batchSize = 16);
		~NeuralNetwork();

	public:
		void addLayer(int width, int height, ActivationFnType type);
		void train(float learningRate = 0.01f);
		float eval(int samplesToTest, bool do_softmax = false);
		std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> inputMat);
		void run();
		void load();
		void save();

		using BatchProvider = std::function<void(
			std::shared_ptr<Matrix>& batchInputMat,		// Pre-allocated input buffer
			std::shared_ptr<Matrix>& batchOutputMat,	// Pre-allocated target buffer
			int batchSize								// Current batch size can be equal or less mat dim
			)>;
		void onTestBatch(const BatchProvider& provider) { m_TestBatchProvider = provider; }
		void onTrainBatch(const BatchProvider& provider) { m_TrainBatchProvider = provider; }

	private:
		void forwardPass(std::shared_ptr<Matrix>& inputBatchMat);
		void targetLayerLossCalc(std::shared_ptr<Matrix>& outputBatchMat);
		void hiddenLayersLossCalc();
		void weightsAndBiasesUpdate(float learningRate);

	private:
		int m_ADAM_Timestep;
		int m_BatchSize;
		BatchProvider m_TestBatchProvider;
		BatchProvider m_TrainBatchProvider;

		std::shared_ptr<Matrix> m_InputBatchMat;
		std::shared_ptr<Matrix> m_OutputBatchMat;

		std::shared_ptr<Shader> 
			m_ForwardPassCompute,
			m_OutputDeltaCompute,
			m_HiddenDeltasCompute,
			m_WeightsCompute,
			m_BiasesCompute;
	public:
		std::vector<std::unique_ptr<NNGL::Layer>> m_Layers;
	};
}
#pragma once

#include "Layer.h"
#include "Shader.h"
#include "Matrix.h"

#include <vector>
#include <queue>
#include <mutex>

namespace NNGL {
	class NeuralNetwork {
	public:
		NeuralNetwork(int batchSize = 16);
		~NeuralNetwork();

		NeuralNetwork(const char *data);
		const char* save();
		size_t getSaveSize() const;
	public:
		std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> inputMat, int use_col_idx = -1);
		std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> inputMat, std::shared_ptr<Matrix> outputMat, float learningRate, int use_col_idx = -1);
		std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, float learningRate);
		std::shared_ptr<Matrix> getCachedOutput() { return m_Layers.back()->m_ActivationMat; }
	public:
		void addLayer(int width, int height, ActivationFnType type);
		void addLayer(const char* data);
		void train(float learningRate = 0.01f);
		float eval(int samplesToTest, bool do_softmax = false);
		void run();

		using BatchProvider = std::function<void(
			std::shared_ptr<Matrix>& batchInputMat,		// Pre-allocated input buffer
			std::shared_ptr<Matrix>& batchOutputMat,	// Pre-allocated target buffer
			int batchSize								// Current batch size can be equal or less mat dim
			)>;
		void onTestBatch(const BatchProvider& provider) { m_TestBatchProvider = provider; }
		void onTrainBatch(const BatchProvider& provider) { m_TrainBatchProvider = provider; }

	private:
		void forwardPass(std::shared_ptr<Matrix>& inputBatchMat, int use_col_idx = -1);
		void targetLayerLossCalc(std::shared_ptr<Matrix>& outputBatchMat);
		void hiddenLayersLossCalc();
		void setTargetLayerLoss(std::shared_ptr<Matrix>& targetLoss);
		void weightsAndBiasesUpdate(std::shared_ptr<Matrix>& inputBatchMat, float learningRate, int use_col_idx = -1);
		void inputGradientCalc();

		void loadShaders();
	private:
		int m_ADAM_Timestep;
		int m_BatchSize;
		BatchProvider m_TestBatchProvider;
		BatchProvider m_TrainBatchProvider;

		std::shared_ptr<Matrix> m_InputBatchMat;
		std::shared_ptr<Matrix> m_OutputBatchMat;
		std::shared_ptr<Matrix> m_CachedInput;
		std::shared_ptr<Matrix> m_InputGradMat;

		std::shared_ptr<Shader> 
			m_ForwardPassCompute,
			m_OutputDeltaCompute,
			m_HiddenDeltasCompute,
			m_WeightsCompute,
			m_BiasesCompute,
			m_InputDeltaCompute;

		std::vector<std::unique_ptr<NNGL::Layer>> m_Layers;
	};
}
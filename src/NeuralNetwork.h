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

	public:
		std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> inputMat);
		std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> inputMat, std::shared_ptr<Matrix> outputMat, float learningRate);
		std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, float learningRate);
		void setTargetLayerLoss(std::shared_ptr<Matrix>& targetLoss);
		std::shared_ptr<Matrix> backward_with_targetloss(std::shared_ptr<Matrix> inputMat, std::shared_ptr<Matrix> targetLoss, float learningRate);
		std::shared_ptr<Matrix> forwardMatOutput;
		std::shared_ptr<Matrix> m_CachedInput; // Cache the input for backward pass
		std::shared_ptr<Matrix> getCachedOutput() { return forwardMatOutput; }
	public:
		void addLayer(int width, int height, ActivationFnType type);
		void train(float learningRate = 0.01f);
		float eval(int samplesToTest, bool do_softmax = false);
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
		void weightsAndBiasesUpdate(std::shared_ptr<Matrix>& inputBatchMat, float learningRate);

		// Memory pooling
		std::shared_ptr<Matrix> getMatrixFromPool(int rows, int cols);
		void returnMatrixToPool(std::shared_ptr<Matrix> matrix);

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
			m_BiasesCompute,
			m_InputDeltaCompute;
		// Memory pool for Matrix objects
		std::queue<std::shared_ptr<Matrix>> m_MatrixPool;
		std::mutex m_PoolMutex;

		GLuint m_InputGradBuffer; // Buffer for input gradient, now owned by NeuralNetwork
	public:
		std::vector<std::unique_ptr<NNGL::Layer>> m_Layers;
		void inputGradientCalc();
	};
}
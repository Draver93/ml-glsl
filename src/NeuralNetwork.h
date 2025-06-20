#pragma once

#include "Layer.h"
#include "Shader.h"

#include <vector>


namespace NNGL {
	class NeuralNetwork {
	public:
		NeuralNetwork(int batchSize = 16) 
			:	m_BatchSize(batchSize),
				m_ADAM_Timestep(0){};

		~NeuralNetwork();

	public:
		void addLayer(int width, int height, ActivationFnType type);
		void train(float learningRate = 0.01f);
		float eval(int samplesToTest, bool do_softmax = false);
		void run();
		void load();
		void save();

		using BatchProvider = std::function<void(
			std::vector<float>& batchInputs,    // Pre-allocated input buffer
			std::vector<float>& batchTargets,   // Pre-allocated target buffer
			int batchSize                       // Current batch size
			)>;
		void onTestBatch(const BatchProvider& provider) { m_TestBatchProvider = provider; }
		void onTrainBatch(const BatchProvider& provider) { m_TrainBatchProvider = provider; }

	private:
		void init();

		void bindTrainingData();
		void forwardPass();
		void targetLayerLossCalc();
		void hiddenLayersLossCalc();
		void weightsAndBiasesUpdate(float learningRate);

	private:
		int m_ADAM_Timestep;
		int m_BatchSize;
		BatchProvider m_TestBatchProvider;
		BatchProvider m_TrainBatchProvider;

		std::vector<float> m_InputVector;
		std::vector<float> m_TargetVector;
		GLuint m_InputBuffer = 0;
		GLuint m_TargetBuffer = 0;

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
#pragma once

#include "NeuralNetwork.h"
#include "Matrix.h"

namespace NNGL {
	class SelfAttention {
	public:
		SelfAttention(int modelDimensions, int headDimensions, int seqLen, bool mask = false);
        ~SelfAttention() {};

        std::shared_ptr<Matrix> forward(const std::shared_ptr<Matrix>& input);
        void backward(const std::shared_ptr<Matrix>& gradOutput, const std::shared_ptr<Matrix>& input);
        void updateWeights(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& gradOutput, float learningRate);

    private:

        int m_ModelDim, m_HeadDim, m_SeqLen;
        bool m_UseMask;

        std::shared_ptr<Shader> 
            m_ForwardPassWeightsCompute, 
            m_ForwardPassScoreCompute, 
            m_GradInputCompute, 
            m_GradWeightCompute, 
            m_WeightsUpdatePassCompute;

        std::shared_ptr<Matrix> m_WeightQueryMat;
        std::shared_ptr<Matrix> m_OutQueryMat;
        std::shared_ptr<Matrix> m_GradQueryInputMat;
        std::shared_ptr<Matrix> m_GradWeightQueryMat;

        std::shared_ptr<Matrix> m_WeightKeyMat;
        std::shared_ptr<Matrix> m_OutKeyMat;
        std::shared_ptr<Matrix> m_GradKeyInputMat;
        std::shared_ptr<Matrix> m_GradWeightKeyMat;

        std::shared_ptr<Matrix> m_WeightValueMat;
        std::shared_ptr<Matrix> m_OutValueMat;
        std::shared_ptr<Matrix> m_GradKeyValueMat;
        std::shared_ptr<Matrix> m_GradWeightValueMat;

        std::shared_ptr<Matrix> m_OutputMat;
	};

}
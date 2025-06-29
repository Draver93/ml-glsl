#pragma once

#include "NeuralNetwork.h"
#include "Matrix.h"

namespace NNGL {
	class AttentionBlock {
	public:
		AttentionBlock(int modelDimensions, int headDimensions, int seqLen, bool mask = false);
        ~AttentionBlock() {};

        //By spec input_kv we can replace kv for cross attention
        std::shared_ptr<Matrix> forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& input_kv = nullptr);

        //return GradInput, GradContext
        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> backward(const std::shared_ptr<Matrix>& gradOutput, const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& context);
 
        void updateWeights(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& gradOutput, float learningRate);
    private:
        void computeProjectionGradients(const std::shared_ptr<Matrix>& gradProjection,
            const std::shared_ptr<Matrix>& cachedInput, const std::shared_ptr<Matrix>& weight,
            std::shared_ptr<Matrix>& gradInput, std::shared_ptr<Matrix>& gradWeight);
    private:

        int m_ModelDim, m_HeadDim, m_SeqLen;
        bool m_UseMask;

        std::shared_ptr<Shader> 
            m_ForwardPassWeightsCompute,
            m_ForwardPassScoreCompute,
            m_ForwardPassOutCompute,
            m_SoftmaxCompute,
            m_BackwardOutputCompute,
            m_BackwardScoresCompute,
            m_WeightsUpdatePassCompute,
            m_BackwardProjectionsCompute,
            m_GradInputCompute,
            m_GradWeightCompute;
        
        std::shared_ptr<Matrix> m_WeightQueryMat;
        std::shared_ptr<Matrix> m_WeightKeyMat;
        std::shared_ptr<Matrix> m_WeightValueMat;

        std::shared_ptr<Matrix> m_OutQueryMat;
        std::shared_ptr<Matrix> m_OutKeyMat;
        std::shared_ptr<Matrix> m_OutValueMat;

        std::shared_ptr<Matrix> m_GradQueryInputMat;
        std::shared_ptr<Matrix> m_GradKeyInputMat;
        std::shared_ptr<Matrix> m_GradValueInputMat;

        std::shared_ptr<Matrix> m_GradWeightQueryMat;
        std::shared_ptr<Matrix> m_GradWeightKeyMat;
        std::shared_ptr<Matrix> m_GradWeightValueMat;

        std::shared_ptr<Matrix> m_OutputMat;

        std::shared_ptr<Matrix> m_CachedQ;
        std::shared_ptr<Matrix> m_CachedK;
        std::shared_ptr<Matrix> m_CachedV;

        std::shared_ptr<Matrix> m_CachedScores;
        std::shared_ptr<Matrix> m_CachedAttentionWeights;

        std::shared_ptr<Matrix> m_CachedInput;
        std::shared_ptr<Matrix> m_CachedContext; //for cross att
        std::shared_ptr<Matrix> m_GradInput;
        std::shared_ptr<Matrix> m_GradContext;

        std::shared_ptr<Matrix> m_GradQ;
        std::shared_ptr<Matrix> m_GradK;
        std::shared_ptr<Matrix> m_GradV;
        std::shared_ptr<Matrix> m_GradScores;
        std::shared_ptr<Matrix> m_GradAttentionWeights;
	};

}
#pragma once

#include "NeuralNetwork.h"
#include "Matrix.h"

namespace NNGL {
	class AttentionBlock {
	public:
		AttentionBlock(int modelDimensions, int numHeads, int seqLen, bool mask = false);
        ~AttentionBlock() {};

        //By spec input_kv we can replace kv for cross attention
        std::shared_ptr<Matrix> forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& input_kv = nullptr);
        
        // New overload with padding mask support
        std::shared_ptr<Matrix> forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& input_kv, const std::vector<int>& paddingMask);
        std::shared_ptr<Matrix> getCachedOutput() const { return m_OutputMat; }
        //return GradInput, GradContext
        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> backward(const std::shared_ptr<Matrix>& gradOutput, const std::shared_ptr<Matrix>& context, float learningRate);
 
        void updateWeights(const std::shared_ptr<Matrix>& weight, const std::shared_ptr<Matrix>& gradWeight, 
                         const std::shared_ptr<Matrix>& adamM, const std::shared_ptr<Matrix>& adamV, float learningRate);
    private:
        void computeProjectionGradients(const std::shared_ptr<Matrix>& gradProjection,
            const std::shared_ptr<Matrix>& cachedInput, const std::shared_ptr<Matrix>& weight,
            std::shared_ptr<Matrix>& gradInput, std::shared_ptr<Matrix>& gradWeight);
        void addMatricesGPU(const std::shared_ptr<Matrix>& A, const std::shared_ptr<Matrix>& B, std::shared_ptr<Matrix>& out);

    private:

        int m_ModelDim, m_NumHeads, m_HeadDim, m_SeqLen;
        bool m_UseMask;
        int m_ADAM_Timestep; // ADAM timestep counter
        
        // Padding mask storage
        std::vector<int> m_PaddingMask;
        GLuint m_PaddingMaskBuffer;


        std::shared_ptr<Shader> 
            m_ForwardPassWeightsCompute,
            m_ForwardPassScoreCompute,
            m_ForwardPassOutCompute,
            m_SoftmaxCompute,
            m_BackwardOutputCompute,
            m_BackwardVCompute,
            m_BackwardScoresCompute,
            m_WeightsUpdatePassCompute,
            m_BackwardProjectionsCompute,
            m_GradInputCompute,
            m_GradWeightCompute,
            m_AddMatrixShader;
        
        std::shared_ptr<Matrix> m_WeightQueryMat;
        std::shared_ptr<Matrix> m_WeightKeyMat;
        std::shared_ptr<Matrix> m_WeightValueMat;

        // ADAM optimization buffers for Q, K, V weights
        std::shared_ptr<Matrix> m_ADAM_M_QueryMat;
        std::shared_ptr<Matrix> m_ADAM_V_QueryMat;
        std::shared_ptr<Matrix> m_ADAM_M_KeyMat;
        std::shared_ptr<Matrix> m_ADAM_V_KeyMat;
        std::shared_ptr<Matrix> m_ADAM_M_ValueMat;
        std::shared_ptr<Matrix> m_ADAM_V_ValueMat;

        std::shared_ptr<Matrix> m_OutQueryMat;
        std::shared_ptr<Matrix> m_OutKeyMat;
        std::shared_ptr<Matrix> m_OutValueMat;

        std::shared_ptr<Matrix> m_GradQueryInputMat;
        std::shared_ptr<Matrix> m_GradKeyInputMat;
        std::shared_ptr<Matrix> m_GradValueInputMat;

        // Cached forward pass values for backprop
        std::shared_ptr<Matrix> m_CachedInput;
        std::shared_ptr<Matrix> m_CachedContext;
        std::shared_ptr<Matrix> m_CachedQ;
        std::shared_ptr<Matrix> m_CachedK;
        std::shared_ptr<Matrix> m_CachedV;
        std::shared_ptr<Matrix> m_CachedScores;
        std::shared_ptr<Matrix> m_CachedAttentionWeights;

        // Will be set if cross-attention

        // Input gradients
        std::shared_ptr<Matrix> m_GradInput;
        std::shared_ptr<Matrix> m_GradContext;

        // Weight gradients
        std::shared_ptr<Matrix> m_GradWeightQueryMat;
        std::shared_ptr<Matrix> m_GradWeightKeyMat;
        std::shared_ptr<Matrix> m_GradWeightValueMat;

        // Intermediate gradients for backprop chain
        std::shared_ptr<Matrix> m_GradQ;
        std::shared_ptr<Matrix> m_GradK;
        std::shared_ptr<Matrix> m_GradV;
        std::shared_ptr<Matrix> m_GradScores;
        std::shared_ptr<Matrix> m_GradAttentionWeights;

        // Output matrix: [seq_len, model_dim] (concatenated heads)
        std::shared_ptr<Matrix> m_OutputMat;
	};
}
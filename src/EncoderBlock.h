#pragma once

#include "AttentionBlock.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "LayerNorm.h"
#include <memory>

namespace NNGL {
    class EncoderBlock {
    private:
        std::unique_ptr<AttentionBlock> m_Attention;
        std::unique_ptr<NeuralNetwork> m_FeedForward;

        // Cache intermediate results for backpropagation
        std::shared_ptr<Matrix> m_CachedInput;
        std::shared_ptr<Matrix> m_CachedAttentionOutput;
        std::shared_ptr<Matrix> m_CachedFfnInput;
        std::unique_ptr<LayerNorm> m_AddNorm1;
        std::unique_ptr<LayerNorm> m_AddNorm2;

    public:
        EncoderBlock(int modelDim, int hiddenDim, int seqLen);

        std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> x);
        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, float learningRate);
    };
}
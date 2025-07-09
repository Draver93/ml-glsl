#pragma once

#include "AttentionBlock.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "LayerNorm.h"
#include <memory>

namespace NNGL {
    class DecoderBlock {
    private:
        std::unique_ptr<AttentionBlock> m_MaskedSelfAttn;   // Masked self-attention
        std::unique_ptr<AttentionBlock> m_CrossAttn;       // Cross-attention (encoder-decoder)
        std::unique_ptr<NeuralNetwork> m_FeedForward;

        std::shared_ptr<Matrix> m_CachedMaskedOut;
        std::shared_ptr<Matrix> m_CachedCrossOut;
        std::shared_ptr<Matrix> m_CachedDecoderInput;
        std::shared_ptr<Matrix> m_CachedEncoderOutput;
        std::unique_ptr<LayerNorm> m_AddNorm1;
        std::unique_ptr<LayerNorm> m_AddNorm2;
        std::unique_ptr<LayerNorm> m_AddNorm3;

    public:
        DecoderBlock(int modelDim, int hiddenDim, int seqLen);

        std::shared_ptr<Matrix> forward(
            std::shared_ptr<Matrix> decoderInput,
            std::shared_ptr<Matrix> encoderOutput
        );

        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, float learningRate);

        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> backwardWithEncoderGrad(
            std::shared_ptr<Matrix> gradOutput, float learningRate);
    };
}
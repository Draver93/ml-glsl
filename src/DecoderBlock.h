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
        
        // New overload with padding mask support
        std::shared_ptr<Matrix> forward(
            std::shared_ptr<Matrix> decoderInput,
            std::shared_ptr<Matrix> encoderOutput,
            const std::vector<int>& decoderPaddingMask,
            const std::vector<int>& encoderPaddingMask
        );

        // New method for decoder-only architectures (like GPT)
        std::shared_ptr<Matrix> forwardDecoderOnly(
            std::shared_ptr<Matrix> decoderInput,
            const std::vector<int>& decoderPaddingMask
        );

        // Backward pass for decoder-only architectures
        std::shared_ptr<Matrix> backwardDecoderOnly(std::shared_ptr<Matrix> gradOutput, float learningRate);

        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, float learningRate);

        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> backwardWithEncoderGrad(
            std::shared_ptr<Matrix> gradOutput, float learningRate);
    };

    class DecoderOnlyBlock {
    private:
        std::unique_ptr<AttentionBlock> m_MaskedSelfAttn;
        std::unique_ptr<NeuralNetwork> m_FeedForward;
        std::unique_ptr<LayerNorm> m_AddNorm1;
        std::unique_ptr<LayerNorm> m_AddNorm2;
        std::shared_ptr<Matrix> m_CachedInput;

    public:
        DecoderOnlyBlock(int modelDim, int hiddenDim, int seqLen);
        
        std::shared_ptr<Matrix> forward(
            std::shared_ptr<Matrix> input,
            const std::vector<int>& paddingMask
        );
        
        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, float learningRate);
    };
}
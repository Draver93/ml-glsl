#include "DecoderBlock.h"
#include "Logger.h"

#include <memory>
#include <iostream>

namespace NNGL {

    DecoderBlock::DecoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int numHeads = 8; // Standard number of heads for transformer

        m_MaskedSelfAttn = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen, /*isMasked=*/true);
        m_FeedForward = std::make_unique<NeuralNetwork>(seqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::LRELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::LRELU);

        m_AddNorm1 = std::make_unique<LayerNorm>(modelDim);
        m_AddNorm2 = std::make_unique<LayerNorm>(modelDim);
    }

    std::shared_ptr<Matrix> DecoderBlock::forward(
        std::shared_ptr<Matrix> input,
        const std::vector<int>& paddingMask
    ) {
        NNGL::Timer timer("DecoderOnlyBlock::forward");
        m_CachedInput = input;

        // 1. Masked self-attention with residual connection
        auto maskedOut = m_MaskedSelfAttn->forward(input, nullptr, paddingMask);
        auto addNorm1Out = m_AddNorm1->forward(maskedOut, input);

        // 2. Feed-forward network with residual connection
        auto mlpOut = m_FeedForward->forward(addNorm1Out);
        auto addNorm2Out = m_AddNorm2->forward(mlpOut, addNorm1Out);

        return addNorm2Out;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        NNGL::Timer timer("DecoderOnlyBlock::backward");

        // Backprop through addNorm2 (main: mlpOut, residual: addNorm1Out)
        m_AddNorm2->backward(gradOutput, m_FeedForward->getCachedOutput(), m_AddNorm1->getCachedOutput());
        auto gradFFNInput = m_FeedForward->backward(m_AddNorm2->getGradInput(), learningRate);
        
        // Backprop through addNorm1 (main: maskedOut, residual: input)
        m_AddNorm1->backward(gradFFNInput, m_MaskedSelfAttn->getCachedOutput(), m_CachedInput);
        auto [gradFromMaskedSelf, maskedGradContext] = m_MaskedSelfAttn->backward(m_AddNorm1->getGradInput(), nullptr, learningRate);
        
        auto result = m_AddNorm1->getGradResidual();

        return result;
    }
}
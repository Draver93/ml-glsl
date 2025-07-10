#include "DecoderBlock.h"
#include "Logger.h"

#include <memory>
#include <iostream>

namespace NNGL {
    DecoderBlock::DecoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int numHeads = 8; // Standard number of heads for transformer

        m_MaskedSelfAttn = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen, /*isMasked=*/true);
        m_CrossAttn = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen); // CrossAttention takes Q, K, V separately

        // Use batch size of 1 for now, will be updated dynamically based on input
        m_FeedForward = std::make_unique<NeuralNetwork>(seqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::LRELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::LRELU);

        // Initialize cache matrices
        m_CachedDecoderInput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedEncoderOutput = std::make_shared<Matrix>(seqLen, modelDim);

        m_AddNorm1 = std::make_unique<LayerNorm>(modelDim);
        m_AddNorm2 = std::make_unique<LayerNorm>(modelDim);
        m_AddNorm3 = std::make_unique<LayerNorm>(modelDim);
    }

    std::shared_ptr<Matrix> DecoderBlock::forward(
        std::shared_ptr<Matrix> decoderInput,
        std::shared_ptr<Matrix> encoderOutput
    ) {
        NNGL::Timer timer("DecoderBlock::forward");
        m_CachedDecoderInput = decoderInput;
        m_CachedEncoderOutput = encoderOutput;
        auto maskedOut = m_MaskedSelfAttn->forward(decoderInput);
        // Use AddNorm output directly; no need to cache at block level
        auto addNorm1Out = m_AddNorm1->forward(maskedOut, decoderInput);
        auto crossOut = m_CrossAttn->forward(addNorm1Out, encoderOutput);
        auto addNorm2Out = m_AddNorm2->forward(crossOut, addNorm1Out);
        auto mlpOut = m_FeedForward->forward(addNorm2Out);
        auto addNorm3Out = m_AddNorm3->forward(mlpOut, addNorm2Out);
        return addNorm3Out;
    }

    std::shared_ptr<Matrix> DecoderBlock::forward(
        std::shared_ptr<Matrix> decoderInput,
        std::shared_ptr<Matrix> encoderOutput,
        const std::vector<int>& decoderPaddingMask,
        const std::vector<int>& encoderPaddingMask
    ) {
        NNGL::Timer timer("DecoderBlock::forward (with mask)");
        m_CachedDecoderInput = decoderInput;
        m_CachedEncoderOutput = encoderOutput;
        auto maskedOut = m_MaskedSelfAttn->forward(decoderInput, nullptr, decoderPaddingMask);
        // Use AddNorm output directly; no need to cache at block level
        auto addNorm1Out = m_AddNorm1->forward(maskedOut, decoderInput);
        auto crossOut = m_CrossAttn->forward(addNorm1Out, encoderOutput, encoderPaddingMask);
        auto addNorm2Out = m_AddNorm2->forward(crossOut, addNorm1Out);
        auto mlpOut = m_FeedForward->forward(addNorm2Out);
        auto addNorm3Out = m_AddNorm3->forward(mlpOut, addNorm2Out);
        return addNorm3Out;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        NNGL::Timer timer("DecoderBlock::backward");
        // Backprop through addNorm3
        m_AddNorm3->backward(gradOutput, m_AddNorm2->getCachedOutput(), m_AddNorm2->getCachedOutput());
        // Backprop through FFN
        auto gradMlpOut = m_AddNorm3->getGradInput();
        auto gradAddNorm2Out = m_AddNorm3->getGradResidual();
        m_FeedForward->backward_with_targetloss(m_AddNorm2->getCachedOutput(), gradMlpOut, learningRate);
        // Backprop through addNorm2
        m_AddNorm2->backward(gradAddNorm2Out, m_AddNorm1->getCachedOutput(), m_AddNorm1->getCachedOutput());
        auto gradCrossOut = m_AddNorm2->getGradInput();
        auto gradAddNorm1Out = m_AddNorm2->getGradResidual();
        // Backprop through Cross-Attention
        auto [gradFromCross, gradContext] = m_CrossAttn->backward(gradCrossOut, m_AddNorm1->getCachedOutput(), m_CachedEncoderOutput, learningRate);
        // Backprop through addNorm1
        m_AddNorm1->backward(gradAddNorm1Out, m_CachedDecoderInput, m_CachedDecoderInput);
        auto gradMaskedOut = m_AddNorm1->getGradInput();
        auto gradDecoderInput = m_AddNorm1->getGradResidual();
        // Backprop through Masked Self-Attention
        auto [gradFromMaskedSelf, maskedGradContext] = m_MaskedSelfAttn->backward(gradMaskedOut, m_CachedDecoderInput, nullptr, learningRate);
        return gradDecoderInput;
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> DecoderBlock::backwardWithEncoderGrad(
        std::shared_ptr<Matrix> gradOutput, float learningRate) {

        NNGL::Timer timer("DecoderBlock::backwardWithEncoderGrad");
        // ---- 1. Backprop through AddNorm3 (final residual + normalization) ----
        m_AddNorm3->backward(gradOutput, m_AddNorm2->getCachedOutput(), m_AddNorm2->getCachedOutput());
        auto gradMlpOut = m_AddNorm3->getGradInput();
        auto gradAddNorm2Out = m_AddNorm3->getGradResidual();
        // Backprop through MLP
        m_FeedForward->backward_with_targetloss(m_AddNorm2->getCachedOutput(), gradMlpOut, learningRate);

        // ---- 2. Backprop through AddNorm2 (cross-attention residual + normalization) ----
        m_AddNorm2->backward(gradAddNorm2Out, m_AddNorm1->getCachedOutput(), m_AddNorm1->getCachedOutput());
        auto gradCrossOut = m_AddNorm2->getGradInput();
        auto gradAddNorm1Out = m_AddNorm2->getGradResidual();
        // Backprop through Cross-Attention
        auto [gradFromCrossQuery, gradFromCrossEncoder] = m_CrossAttn->backward(gradCrossOut, m_AddNorm1->getCachedOutput(), m_CachedEncoderOutput, learningRate);

        // ---- 3. Backprop through AddNorm1 (masked self-attention residual + normalization) ----
        m_AddNorm1->backward(gradFromCrossQuery, m_CachedDecoderInput, m_CachedDecoderInput);
        auto gradMaskedOut = m_AddNorm1->getGradInput();
        auto gradDecoderInput = m_AddNorm1->getGradResidual();
        // Backprop through Masked Self-Attention
        auto [gradFromMaskedSelf, gradFromMaskedEncoder] = m_MaskedSelfAttn->backward(gradMaskedOut, m_CachedDecoderInput, nullptr, learningRate);

        // Return BOTH gradients: decoder input gradient AND encoder output gradient
        return std::make_pair(gradFromMaskedSelf, gradFromCrossEncoder);
    }
}
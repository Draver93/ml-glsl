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
        auto addNorm1Out = m_AddNorm1->forward(maskedOut, decoderInput);

        auto crossOut = m_CrossAttn->forward(addNorm1Out, encoderOutput, encoderPaddingMask);
        auto addNorm2Out = m_AddNorm2->forward(crossOut, addNorm1Out);

        auto mlpOut = m_FeedForward->forward(addNorm2Out);
        auto addNorm3Out = m_AddNorm3->forward(mlpOut, addNorm2Out);

        return addNorm3Out;
    }

    std::shared_ptr<Matrix> DecoderBlock::forwardDecoderOnly(
        std::shared_ptr<Matrix> decoderInput,
        const std::vector<int>& decoderPaddingMask
    ) {
        NNGL::Timer timer("DecoderBlock::forwardDecoderOnly");
        m_CachedDecoderInput = decoderInput;
        // For decoder-only, we don't have encoder output
        m_CachedEncoderOutput = nullptr;

        // 1. Masked self-attention
        auto maskedOut = m_MaskedSelfAttn->forward(decoderInput, nullptr, decoderPaddingMask);
        auto addNorm1Out = m_AddNorm1->forward(maskedOut, decoderInput);

        // 2. Skip cross-attention (decoder-only architecture)
        // Just pass through the addNorm1Out as if cross-attention didn't change it
        auto addNorm2Out = m_AddNorm2->forward(addNorm1Out, addNorm1Out);

        // 3. Feed-forward network
        auto mlpOut = m_FeedForward->forward(addNorm2Out);
        auto addNorm3Out = m_AddNorm3->forward(mlpOut, addNorm2Out);

        return addNorm3Out;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        NNGL::Timer timer("DecoderBlock::backward");

        // Backprop through addNorm3 (main: mlpOut, residual: addNorm2Out)
        m_AddNorm3->backward(gradOutput, m_FeedForward->getCachedOutput(), m_AddNorm2->getCachedOutput());
        auto gradFFNInput = m_FeedForward->backward(m_AddNorm3->getGradInput(), learningRate);
        
        // Backprop through addNorm2 (main: crossOut, residual: addNorm1Out)
        m_AddNorm2->backward(gradFFNInput, m_CrossAttn->getCachedOutput(), m_AddNorm1->getCachedOutput());
        auto [gradFromCross, gradContext] = m_CrossAttn->backward(m_AddNorm2->getGradInput(), m_CachedEncoderOutput, learningRate);
        
        // Backprop through addNorm1 (main: maskedOut, residual: decoderInput)
        m_AddNorm1->backward(gradContext, m_MaskedSelfAttn->getCachedOutput(), m_CachedDecoderInput);
        auto [gradFromMaskedSelf, maskedGradContext] = m_MaskedSelfAttn->backward(m_AddNorm1->getGradInput(), nullptr, learningRate);
        
        auto result = m_AddNorm1->getGradResidual();

        return result;
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> DecoderBlock::backwardWithEncoderGrad(
        std::shared_ptr<Matrix> gradOutput, float learningRate) {
        NNGL::Timer timer("DecoderBlock::backwardWithEncoderGrad");
        
        // ---- 1. Backprop through AddNorm3 (final residual + normalization) ----
        m_AddNorm3->backward(gradOutput, m_AddNorm2->getCachedOutput(), m_AddNorm2->getCachedOutput());
        auto gradFFNInput = m_FeedForward->backward(m_AddNorm3->getGradInput(), learningRate);

        // ---- 2. Backprop through AddNorm2 (cross-attention residual + normalization) ----
        m_AddNorm2->backward(gradFFNInput, m_AddNorm1->getCachedOutput(), m_AddNorm1->getCachedOutput());
        auto [gradFromCrossQuery, gradFromCrossEncoder] = m_CrossAttn->backward(m_AddNorm2->getGradInput(), m_CachedEncoderOutput, learningRate);

        // ---- 3. Backprop through AddNorm1 (masked self-attention residual + normalization) ----
        m_AddNorm1->backward(gradFromCrossQuery, m_CachedDecoderInput, m_CachedDecoderInput);
        auto [gradFromMaskedSelf, gradFromMaskedEncoder] = m_MaskedSelfAttn->backward(m_AddNorm1->getGradInput(), nullptr, learningRate);

        // Return BOTH gradients: decoder input gradient AND encoder output gradient
        return std::make_pair(gradFromMaskedSelf, gradFromCrossEncoder);
    }

    std::shared_ptr<Matrix> DecoderBlock::backwardDecoderOnly(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        NNGL::Timer timer("DecoderBlock::backwardDecoderOnly");

        // Backprop through addNorm3 (main: mlpOut, residual: addNorm2Out)
        m_AddNorm3->backward(gradOutput, m_FeedForward->getCachedOutput(), m_AddNorm2->getCachedOutput());
        auto gradFFNInput = m_FeedForward->backward(m_AddNorm3->getGradInput(), learningRate);
        
        // Backprop through addNorm2 (for decoder-only, main and residual are the same: addNorm1Out)
        m_AddNorm2->backward(gradFFNInput, m_AddNorm1->getCachedOutput(), m_AddNorm1->getCachedOutput());
        // Skip cross-attention backward pass since we didn't do cross-attention in forward
        
        // Backprop through addNorm1 (main: maskedOut, residual: decoderInput)
        m_AddNorm1->backward(m_AddNorm2->getGradInput(), m_MaskedSelfAttn->getCachedOutput(), m_CachedDecoderInput);
        auto [gradFromMaskedSelf, maskedGradContext] = m_MaskedSelfAttn->backward(m_AddNorm1->getGradInput(), nullptr, learningRate);
        
        auto result = m_AddNorm1->getGradResidual();

        return result;
    }

    DecoderOnlyBlock::DecoderOnlyBlock(int modelDim, int hiddenDim, int seqLen) {
        int numHeads = 8; // Standard number of heads for transformer

        m_MaskedSelfAttn = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen, /*isMasked=*/true);
        m_FeedForward = std::make_unique<NeuralNetwork>(seqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::LRELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::LRELU);

        m_AddNorm1 = std::make_unique<LayerNorm>(modelDim);
        m_AddNorm2 = std::make_unique<LayerNorm>(modelDim);
    }

    std::shared_ptr<Matrix> DecoderOnlyBlock::forward(
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

    std::shared_ptr<Matrix> DecoderOnlyBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
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
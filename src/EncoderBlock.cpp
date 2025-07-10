#include "EncoderBlock.h"
#include "Logger.h"

#include <memory>
#include <iostream>

namespace NNGL {
    EncoderBlock::EncoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int numHeads = 8; // Standard number of heads for transformer
        m_Attention = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen, false); // No masking for encoder

        m_FeedForward = std::make_unique<NeuralNetwork>(seqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::LRELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::LRELU);

        // Initialize cache matrices
        m_CachedInput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedAttentionOutput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedFfnInput = std::make_shared<Matrix>(seqLen, modelDim);
        m_AddNorm1 = std::make_unique<LayerNorm>(modelDim);
        m_AddNorm2 = std::make_unique<LayerNorm>(modelDim);
    }

    std::shared_ptr<Matrix> EncoderBlock::forward(std::shared_ptr<Matrix> x) {
        NNGL::Timer timer("EncoderBlock::forward");
        // Dimension check before assignment
        if (m_CachedInput && (m_CachedInput->rows != x->rows || m_CachedInput->cols != x->cols))
            throw std::runtime_error("EncoderBlock: m_CachedInput dimension mismatch");
        m_CachedInput = x;
        std::shared_ptr<Matrix> attentionOutput = m_Attention->forward(x);
        auto addNorm1Out = m_AddNorm1->forward(attentionOutput, x);
        if (m_CachedAttentionOutput && (m_CachedAttentionOutput->rows != addNorm1Out->rows || m_CachedAttentionOutput->cols != addNorm1Out->cols))
            throw std::runtime_error("EncoderBlock: m_CachedAttentionOutput dimension mismatch");
        m_CachedAttentionOutput = addNorm1Out;
        if (m_CachedFfnInput && (m_CachedFfnInput->rows != addNorm1Out->rows || m_CachedFfnInput->cols != addNorm1Out->cols))
            throw std::runtime_error("EncoderBlock: m_CachedFfnInput dimension mismatch");
        m_CachedFfnInput = addNorm1Out;
        std::shared_ptr<Matrix> mlpOut = m_FeedForward->forward(addNorm1Out);
        auto addNorm2Out = m_AddNorm2->forward(mlpOut, addNorm1Out);
        return addNorm2Out;
    }

    std::shared_ptr<Matrix> EncoderBlock::forward(std::shared_ptr<Matrix> x, const std::vector<int>& paddingMask) {
        NNGL::Timer timer("EncoderBlock::forward (with mask)");
        // Dimension check before assignment
        if (m_CachedInput && (m_CachedInput->rows != x->rows || m_CachedInput->cols != x->cols))
            throw std::runtime_error("EncoderBlock: m_CachedInput dimension mismatch");
        m_CachedInput = x;
        std::shared_ptr<Matrix> attentionOutput = m_Attention->forward(x, nullptr, paddingMask);
        auto addNorm1Out = m_AddNorm1->forward(attentionOutput, x);
        if (m_CachedAttentionOutput && (m_CachedAttentionOutput->rows != addNorm1Out->rows || m_CachedAttentionOutput->cols != addNorm1Out->cols))
            throw std::runtime_error("EncoderBlock: m_CachedAttentionOutput dimension mismatch");
        m_CachedAttentionOutput = addNorm1Out;
        if (m_CachedFfnInput && (m_CachedFfnInput->rows != addNorm1Out->rows || m_CachedFfnInput->cols != addNorm1Out->cols))
            throw std::runtime_error("EncoderBlock: m_CachedFfnInput dimension mismatch");
        m_CachedFfnInput = addNorm1Out;
        std::shared_ptr<Matrix> mlpOut = m_FeedForward->forward(addNorm1Out);
        auto addNorm2Out = m_AddNorm2->forward(mlpOut, addNorm1Out);
        return addNorm2Out;
    }

    std::shared_ptr<Matrix> EncoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        NNGL::Timer timer("EncoderBlock::backward");
        // Backprop through addNorm2
        std::shared_ptr<Matrix> gradMlpOut, gradAddNorm1Out, gradGamma2, gradBeta2;
        m_AddNorm2->backward(gradOutput, m_AddNorm1->getCachedOutput(), m_AddNorm1->getCachedOutput(), gradMlpOut, gradAddNorm1Out, gradGamma2, gradBeta2);
        // Backprop through FFN
        auto gradFromFfn = m_FeedForward->backward_with_targetloss(m_AddNorm1->getCachedOutput(), gradMlpOut, learningRate);
        // Backprop through addNorm1
        std::shared_ptr<Matrix> gradAttentionOut, gradInput, gradGamma1, gradBeta1;
        m_AddNorm1->backward(gradAddNorm1Out, m_CachedInput, m_CachedInput, gradAttentionOut, gradInput, gradGamma1, gradBeta1);
        // Backprop through Attention
        auto [gradFromAttention, gradContext] = m_Attention->backward(gradAttentionOut, m_CachedInput, nullptr);
        return gradInput;
    }
}
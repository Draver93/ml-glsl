#include "EncoderBlock.h"
#include <memory>

namespace NNGL {
    EncoderBlock::EncoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int numHeads = 8; // Standard number of heads for transformer
        m_Attention = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen, false); // No masking for encoder

        m_FeedForward = std::make_unique<NeuralNetwork>(seqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::RELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::RELU);

        // Initialize cache matrices
        m_CachedInput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedAttentionOutput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedFfnInput = std::make_shared<Matrix>(seqLen, modelDim);
    }

    std::shared_ptr<Matrix> EncoderBlock::forward(std::shared_ptr<Matrix> x) {
        // Cache input for backpropagation
        m_CachedInput->copyFrom(x);

        // Self-m_Attention
        std::shared_ptr<Matrix> attentionOutput = m_Attention->forward(x);
        attentionOutput->add(*x);  // First residual connection

        // Cache m_Attention output (after residual)
        m_CachedAttentionOutput->copyFrom(attentionOutput);
        m_CachedFfnInput->copyFrom(attentionOutput);

        // Feed-forward network
        std::shared_ptr<Matrix> mlpOut = m_FeedForward->forward(attentionOutput);
        mlpOut->add(*attentionOutput); // Second residual connection

        return mlpOut;
    }

    std::shared_ptr<Matrix> EncoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        // ---- 1. Backprop through second residual connection and FFN ----
        // gradOutput flows to both the FFN and the residual connection
        auto gradFfnInputFromResidual = std::make_shared<Matrix>(*gradOutput);
        auto gradFfnInput = std::make_shared<Matrix>(*gradOutput);

        // Backprop through m_FeedForward network
        auto gradFromFfn = m_FeedForward->backward_with_targetloss(m_CachedFfnInput, gradFfnInput, learningRate);

        // Add gradient from residual connection
        gradFromFfn->add(*gradFfnInputFromResidual);

        // ---- 2. Backprop through first residual connection and self-m_Attention ----
        // gradFromFfn flows to both the m_Attention and the residual connection
        auto gradInputFromResidual = std::make_shared<Matrix>(*gradFromFfn);
        auto gradAttentionInput = std::make_shared<Matrix>(*gradFromFfn);

        // Backprop through self-m_Attention (no context for encoder self-m_Attention)
        auto [ gradFromAttention, gradContext ] = m_Attention->backward(gradAttentionInput, m_CachedInput, nullptr);

        // Add gradient from residual connection
        gradFromAttention->add(*gradInputFromResidual);

        return gradFromAttention;
    }
} 
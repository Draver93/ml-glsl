#include "EncoderBlock.h"
#include "Logger.h"

#include <memory>
#include <iostream>

namespace NNGL {
    EncoderBlock::EncoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int numHeads = 8; // Standard number of heads for transformer
        m_Attention = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen, false); // No masking for encoder

        m_FeedForward = std::make_unique<NeuralNetwork>(seqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::RELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::RELU);

        // Initialize layer normalization layers
        m_LayerNorm1 = std::make_unique<LayerNorm>(modelDim);  // After attention
        m_LayerNorm2 = std::make_unique<LayerNorm>(modelDim);  // After feed-forward

        // Initialize cache matrices
        m_CachedInput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedAttentionOutput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedFfnInput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedNorm1Output = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedNorm2Output = std::make_shared<Matrix>(seqLen, modelDim);
    }

    std::shared_ptr<Matrix> EncoderBlock::forward(std::shared_ptr<Matrix> x) {
        // Cache input for backpropagation
        m_CachedInput->copyFrom(x);

        // Self-Attention
        std::shared_ptr<Matrix> attentionOutput = m_Attention->forward(x);
        
        // Debug: Check dimensions before first residual connection
        LOG_TRACE("  EncoderBlock: Input x: [" + std::to_string(x->rows) + "," + std::to_string(x->cols) + "]");
        LOG_TRACE("  EncoderBlock: Attention output: [" + std::to_string(attentionOutput->rows) + "," + std::to_string(attentionOutput->cols) + "]");
        
        // Add & Norm: Residual connection + Layer Normalization
        attentionOutput->add(*x);  // First residual connection
        std::shared_ptr<Matrix> norm1Output = m_LayerNorm1->forward(attentionOutput);
        m_CachedNorm1Output->copyFrom(norm1Output);

        // Cache attention output (after residual + norm) for feed-forward
        m_CachedAttentionOutput->copyFrom(attentionOutput);
        m_CachedFfnInput->copyFrom(norm1Output);

        // Feed-forward network
        std::shared_ptr<Matrix> mlpOut = m_FeedForward->forward(norm1Output);
        
        // Debug: Check dimensions before second residual connection
        LOG_TRACE("  EncoderBlock: MLP output: [" + std::to_string(mlpOut->rows) + "," + std::to_string(mlpOut->cols) + "]");
        LOG_TRACE("  EncoderBlock: Norm1 output (for residual): [" + std::to_string(norm1Output->rows) + "," + std::to_string(norm1Output->cols) + "]");
        
        // Add & Norm: Residual connection + Layer Normalization
        mlpOut->add(*norm1Output); // Second residual connection
        std::shared_ptr<Matrix> norm2Output = m_LayerNorm2->forward(mlpOut);
        m_CachedNorm2Output->copyFrom(norm2Output);

        return norm2Output;
    }

    std::shared_ptr<Matrix> EncoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        // ---- 1. Backprop through second LayerNorm and residual connection ----
        auto gradFromNorm2 = m_LayerNorm2->backward(gradOutput, learningRate);
        
        // Split gradient for residual connection and FFN
        auto gradFfnInputFromResidual = std::make_shared<Matrix>(*gradFromNorm2);
        auto gradFfnInput = std::make_shared<Matrix>(*gradFromNorm2);

        // Backprop through Feed-Forward network
        auto gradFromFfn = m_FeedForward->backward_with_targetloss(m_CachedFfnInput, gradFfnInput, learningRate);

        // Add gradient from residual connection
        gradFromFfn->add(*gradFfnInputFromResidual);

        // ---- 2. Backprop through first LayerNorm and residual connection ----
        auto gradFromNorm1 = m_LayerNorm1->backward(gradFromFfn, learningRate);
        
        // Split gradient for residual connection and attention
        auto gradInputFromResidual = std::make_shared<Matrix>(*gradFromNorm1);
        auto gradAttentionInput = std::make_shared<Matrix>(*gradFromNorm1);

        // Backprop through Self-Attention (no context for encoder self-attention)
        auto [ gradFromAttention, gradContext ] = m_Attention->backward(gradAttentionInput, m_CachedInput, nullptr);

        // Add gradient from residual connection
        gradFromAttention->add(*gradInputFromResidual);

        return gradFromAttention;
    }
} 
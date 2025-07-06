#include "DecoderBlock.h"
#include "Logger.h"

#include <memory>
#include <iostream>

namespace NNGL {
    DecoderBlock::DecoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int numHeads = 8; // Standard number of heads for transformer

        m_MaskedSelfAttn = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen, /*isMasked=*/true);
        m_CrossAttn = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen); // CrossAttention takes Q, K, V separately

        m_FeedForward = std::make_unique<NeuralNetwork>(seqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::RELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::RELU);

        // Initialize layer normalization layers
        m_LayerNorm1 = std::make_unique<LayerNorm>(modelDim);  // After masked self-attention
        m_LayerNorm2 = std::make_unique<LayerNorm>(modelDim);  // After cross-attention
        m_LayerNorm3 = std::make_unique<LayerNorm>(modelDim);  // After feed-forward

        // Initialize cache matrices
        m_CachedMaskedOut = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedCrossOut = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedDecoderInput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedEncoderOutput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedNorm1Output = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedNorm2Output = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedNorm3Output = std::make_shared<Matrix>(seqLen, modelDim);
    }

    std::shared_ptr<Matrix> DecoderBlock::forward(
        std::shared_ptr<Matrix> decoderInput,
        std::shared_ptr<Matrix> encoderOutput
    ) {
        // Cache inputs for backprop
        m_CachedDecoderInput->copyFrom(decoderInput);
        m_CachedEncoderOutput->copyFrom(encoderOutput);

        // 1. Masked Self-Attention + Add & Norm
        auto maskedOut = m_MaskedSelfAttn->forward(decoderInput);
        
        // Debug: Check dimensions before first residual connection
        LOG_TRACE("  DecoderBlock: Decoder input: [" + std::to_string(decoderInput->rows) + "," + std::to_string(decoderInput->cols) + "]");
        LOG_TRACE("  DecoderBlock: Masked attention output: [" + std::to_string(maskedOut->rows) + "," + std::to_string(maskedOut->cols) + "]");
        
        maskedOut->add(*decoderInput);  // first residual
        std::shared_ptr<Matrix> norm1Output = m_LayerNorm1->forward(maskedOut);
        m_CachedNorm1Output->copyFrom(norm1Output);

        // 2. Cross-Attention + Add & Norm
        auto crossOut = m_CrossAttn->forward(norm1Output, encoderOutput);
        
        // Debug: Check dimensions before second residual connection
        LOG_TRACE("  DecoderBlock: Cross attention output: [" + std::to_string(crossOut->rows) + "," + std::to_string(crossOut->cols) + "]");
        LOG_TRACE("  DecoderBlock: Norm1 output (for residual): [" + std::to_string(norm1Output->rows) + "," + std::to_string(norm1Output->cols) + "]");
        
        crossOut->add(*norm1Output);      // second residual
        std::shared_ptr<Matrix> norm2Output = m_LayerNorm2->forward(crossOut);
        m_CachedNorm2Output->copyFrom(norm2Output);

        // 3. Feed-Forward + Add & Norm
        auto mlpOut = m_FeedForward->forward(norm2Output);
        
        // Debug: Check dimensions before third residual connection
        LOG_TRACE("  DecoderBlock: MLP output: [" + std::to_string(mlpOut->rows) + "," + std::to_string(mlpOut->cols) + "]");
        LOG_TRACE("  DecoderBlock: Norm2 output (for residual): [" + std::to_string(norm2Output->rows) + "," + std::to_string(norm2Output->cols) + "]");
        
        mlpOut->add(*norm2Output);         // third residual
        std::shared_ptr<Matrix> norm3Output = m_LayerNorm3->forward(mlpOut);
        m_CachedNorm3Output->copyFrom(norm3Output);

        return norm3Output;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        // ---- 1. Backprop through final LayerNorm and residual connection ----
        auto gradFromNorm3 = m_LayerNorm3->backward(gradOutput, learningRate);
        
        // Split gradient for residual connection and MLP
        auto gradCrossOutFromResidual = std::make_shared<Matrix>(*gradFromNorm3);
        auto gradMlpInput = std::make_shared<Matrix>(*gradFromNorm3);

        auto gradFromMlp = m_FeedForward->backward_with_targetloss(m_CachedNorm2Output, gradMlpInput, learningRate);
        gradFromMlp->add(*gradCrossOutFromResidual);

        // ---- 2. Backprop through second LayerNorm and residual connection ----
        auto gradFromNorm2 = m_LayerNorm2->backward(gradFromMlp, learningRate);
        
        // Split gradient for residual connection and Cross-Attention
        auto gradMaskedOutFromResidual = std::make_shared<Matrix>(*gradFromNorm2);
        auto gradCrossInput = std::make_shared<Matrix>(*gradFromNorm2);

        auto [gradFromCross, gradContext] = m_CrossAttn->backward(gradCrossInput, m_CachedNorm1Output, m_CachedEncoderOutput);

        gradFromCross->add(*gradMaskedOutFromResidual);

        // ---- 3. Backprop through first LayerNorm and residual connection ----
        auto gradFromNorm1 = m_LayerNorm1->backward(gradFromCross, learningRate);
        
        // Split gradient for residual connection and Masked Self-Attention
        auto gradDecoderInputFromResidual = std::make_shared<Matrix>(*gradFromNorm1);
        auto gradMaskedInput = std::make_shared<Matrix>(*gradFromNorm1);

        auto [gradFromMaskedSelf, maskedGradContext] = m_MaskedSelfAttn->backward(gradMaskedInput, m_CachedDecoderInput, nullptr);
        gradFromMaskedSelf->add(*gradDecoderInputFromResidual);

        return gradFromMaskedSelf;
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> DecoderBlock::backwardWithEncoderGrad(
        std::shared_ptr<Matrix> gradOutput, float learningRate) {

        // ---- 1. Backprop through final LayerNorm and residual connection ----
        auto gradFromNorm3 = m_LayerNorm3->backward(gradOutput, learningRate);
        
        // Split gradient for residual connection and MLP
        auto gradCrossOutFromResidual = std::make_shared<Matrix>(*gradFromNorm3);
        auto gradMlpInput = std::make_shared<Matrix>(*gradFromNorm3);

        auto gradFromMlp = m_FeedForward->backward_with_targetloss(m_CachedNorm2Output, gradMlpInput, learningRate);
        gradFromMlp->add(*gradCrossOutFromResidual);

        // ---- 2. Backprop through second LayerNorm and residual connection ----
        auto gradFromNorm2 = m_LayerNorm2->backward(gradFromMlp, learningRate);
        
        // Split gradient for residual connection and Cross-Attention
        auto gradMaskedOutFromResidual = std::make_shared<Matrix>(*gradFromNorm2);
        auto gradCrossInput = std::make_shared<Matrix>(*gradFromNorm2);

        auto [gradFromCrossQuery, gradFromCrossEncoder] = m_CrossAttn->backward(gradCrossInput, m_CachedNorm1Output, m_CachedEncoderOutput);

        gradFromCrossQuery->add(*gradMaskedOutFromResidual);

        // ---- 3. Backprop through first LayerNorm and residual connection ----
        auto gradFromNorm1 = m_LayerNorm1->backward(gradFromCrossQuery, learningRate);
        
        // Split gradient for residual connection and Masked Self-Attention
        auto gradDecoderInputFromResidual = std::make_shared<Matrix>(*gradFromNorm1);
        auto gradMaskedInput = std::make_shared<Matrix>(*gradFromNorm1);

        auto [gradFromMaskedSelf, gradFromMaskedEncoder] = m_MaskedSelfAttn->backward(gradMaskedInput, m_CachedDecoderInput, nullptr);
        gradFromMaskedSelf->add(*gradDecoderInputFromResidual);

        // Return BOTH gradients: decoder input gradient AND encoder output gradient
        return std::make_pair(gradFromMaskedSelf, gradFromCrossEncoder);
    }
} 
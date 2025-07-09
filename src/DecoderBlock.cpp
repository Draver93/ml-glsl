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
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::RELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::RELU);

        // Initialize cache matrices
        m_CachedMaskedOut = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedCrossOut = std::make_shared<Matrix>(seqLen, modelDim);
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
        m_CachedDecoderInput->copyFrom(decoderInput);
        m_CachedEncoderOutput->copyFrom(encoderOutput);
        auto maskedOut = m_MaskedSelfAttn->forward(decoderInput);
        auto addNorm1Out = m_AddNorm1->forward(maskedOut, decoderInput);
        m_CachedMaskedOut->copyFrom(addNorm1Out);
        auto crossOut = m_CrossAttn->forward(addNorm1Out, encoderOutput);
        auto addNorm2Out = m_AddNorm2->forward(crossOut, addNorm1Out);
        m_CachedCrossOut->copyFrom(addNorm2Out);
        auto mlpOut = m_FeedForward->forward(addNorm2Out);
        auto addNorm3Out = m_AddNorm3->forward(mlpOut, addNorm2Out);
        return addNorm3Out;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        // Backprop through addNorm3
        std::shared_ptr<Matrix> gradMlpOut, gradAddNorm2Out, gradGamma3, gradBeta3;
        m_AddNorm3->backward(gradOutput, m_CachedCrossOut, m_CachedCrossOut, gradMlpOut, gradAddNorm2Out, gradGamma3, gradBeta3);
        // Backprop through FFN
        auto gradFromMlp = m_FeedForward->backward_with_targetloss(m_CachedCrossOut, gradMlpOut, learningRate);
        // Backprop through addNorm2
        std::shared_ptr<Matrix> gradCrossOut, gradAddNorm1Out, gradGamma2, gradBeta2;
        m_AddNorm2->backward(gradAddNorm2Out, m_CachedCrossOut, m_CachedMaskedOut, gradCrossOut, gradAddNorm1Out, gradGamma2, gradBeta2);
        // Backprop through Cross-Attention
        auto [gradFromCross, gradContext] = m_CrossAttn->backward(gradCrossOut, m_CachedMaskedOut, m_CachedEncoderOutput);
        // Backprop through addNorm1
        std::shared_ptr<Matrix> gradMaskedOut, gradDecoderInput, gradGamma1, gradBeta1;
        m_AddNorm1->backward(gradAddNorm1Out, m_CachedMaskedOut, m_CachedDecoderInput, gradMaskedOut, gradDecoderInput, gradGamma1, gradBeta1);
        // Backprop through Masked Self-Attention
        auto [gradFromMaskedSelf, maskedGradContext] = m_MaskedSelfAttn->backward(gradMaskedOut, m_CachedDecoderInput, nullptr);
        return gradDecoderInput;
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> DecoderBlock::backwardWithEncoderGrad(
        std::shared_ptr<Matrix> gradOutput, float learningRate) {

        // ---- 1. Backprop through final residual connection and MLP ----
        auto gradCrossOutFromResidual = std::make_shared<Matrix>(*gradOutput);
        auto gradMlpInput = std::make_shared<Matrix>(*gradOutput);

        auto gradFromMlp = m_FeedForward->backward_with_targetloss(m_CachedCrossOut, gradMlpInput, learningRate);
        gradFromMlp->add(*gradCrossOutFromResidual);

        // ---- 2. Backprop through second residual connection and Cross-Attention ----
        auto gradMaskedOutFromResidual = std::make_shared<Matrix>(*gradFromMlp);
        auto gradCrossInput = std::make_shared<Matrix>(*gradFromMlp);


        auto [gradFromCrossQuery, gradFromCrossEncoder] = m_CrossAttn->backward(gradCrossInput, m_CachedMaskedOut, m_CachedEncoderOutput);

        gradFromCrossQuery->add(*gradMaskedOutFromResidual);

        // ---- 3. Backprop through first residual connection and Masked Self-Attention ----
        auto gradDecoderInputFromResidual = std::make_shared<Matrix>(*gradFromCrossQuery);
        auto gradMaskedInput = std::make_shared<Matrix>(*gradFromCrossQuery);

        auto [gradFromMaskedSelf, gradFromMaskedEncoder] = m_MaskedSelfAttn->backward(gradMaskedInput, m_CachedDecoderInput, nullptr);
        gradFromMaskedSelf->add(*gradDecoderInputFromResidual);

        // Return BOTH gradients: decoder input gradient AND encoder output gradient
        return std::make_pair(gradFromMaskedSelf, gradFromCrossEncoder);
    }
}
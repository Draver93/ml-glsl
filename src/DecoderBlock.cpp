#include "DecoderBlock.h"
#include <memory>

namespace NNGL {
    DecoderBlock::DecoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int numHeads = 8; // Standard number of heads for transformer

        m_MaskedSelfAttn = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen, /*isMasked=*/true);
        m_CrossAttn = std::make_unique<AttentionBlock>(modelDim, numHeads, seqLen); // CrossAttention takes Q, K, V separately

        m_FeedForward = std::make_unique<NeuralNetwork>(seqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, NNGL::ActivationFnType::RELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, NNGL::ActivationFnType::RELU);

        // Initialize cache matrices
        m_CachedMaskedOut = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedCrossOut = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedDecoderInput = std::make_shared<Matrix>(seqLen, modelDim);
        m_CachedEncoderOutput = std::make_shared<Matrix>(seqLen, modelDim);
    }

    std::shared_ptr<Matrix> DecoderBlock::forward(
        std::shared_ptr<Matrix> decoderInput,
        std::shared_ptr<Matrix> encoderOutput
    ) {
        // Cache inputs for backprop
        m_CachedDecoderInput->copyFrom(decoderInput);
        m_CachedEncoderOutput->copyFrom(encoderOutput);

        auto maskedOut = m_MaskedSelfAttn->forward(decoderInput);
        maskedOut->add(*decoderInput);  // first residual
        m_CachedMaskedOut->copyFrom(maskedOut);  // cache this intermediate result

        auto crossOut = m_CrossAttn->forward(maskedOut, encoderOutput);
        crossOut->add(*maskedOut);      // second residual
        m_CachedCrossOut->copyFrom(crossOut);  // cache this intermediate result

        auto mlpOut = m_FeedForward->forward(crossOut);
        mlpOut->add(*crossOut);         // third residual

        return mlpOut;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        // ---- 1. Backprop through final residual connection and MLP ----
        auto gradCrossOutFromResidual = std::make_shared<Matrix>(*gradOutput);
        auto gradMlpInput = std::make_shared<Matrix>(*gradOutput);

        auto gradFromMlp = m_FeedForward->backward_with_targetloss(m_CachedCrossOut, gradMlpInput, learningRate);
        gradFromMlp->add(*gradCrossOutFromResidual);

        // ---- 2. Backprop through second residual connection and Cross-Attention ----
        auto gradMaskedOutFromResidual = std::make_shared<Matrix>(*gradFromMlp);
        auto gradCrossInput = std::make_shared<Matrix>(*gradFromMlp);

        auto [gradFromCross, gradContext] = m_CrossAttn->backward(gradCrossInput, m_CachedMaskedOut, m_CachedEncoderOutput);

        gradFromCross->add(*gradMaskedOutFromResidual);

        // ---- 3. Backprop through first residual connection and Masked Self-Attention ----
        auto gradDecoderInputFromResidual = std::make_shared<Matrix>(*gradFromCross);
        auto gradMaskedInput = std::make_shared<Matrix>(*gradFromCross);

        auto [gradFromMaskedSelf, maskedGradContext] = m_MaskedSelfAttn->backward(gradMaskedInput, m_CachedDecoderInput, nullptr);
        gradFromMaskedSelf->add(*gradDecoderInputFromResidual);

        return gradFromMaskedSelf;
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
#include "EncoderBlock.h"
#include <memory>

namespace NNGL {
    EncoderBlock::EncoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int headDim = modelDim; // for simplicity
        attention = std::make_unique<AttentionBlock>(modelDim, headDim, seqLen, false); // No masking for encoder

        feedForward = std::make_unique<NeuralNetwork>(seqLen);
        feedForward->addLayer(headDim, hiddenDim, NNGL::ActivationFnType::RELU);
        feedForward->addLayer(hiddenDim, headDim, NNGL::ActivationFnType::RELU);

        // Initialize cache matrices
        cachedInput = std::make_shared<Matrix>(seqLen, modelDim);
        cachedAttentionOutput = std::make_shared<Matrix>(seqLen, modelDim);
        cachedFfnInput = std::make_shared<Matrix>(seqLen, modelDim);
    }

    std::shared_ptr<Matrix> EncoderBlock::forward(std::shared_ptr<Matrix> x) {
        // Cache input for backpropagation
        cachedInput->copyFrom(x);

        // Self-attention
        std::shared_ptr<Matrix> attentionOutput = attention->forward(x);
        attentionOutput->add(*x);  // First residual connection

        // Cache attention output (after residual)
        cachedAttentionOutput->copyFrom(attentionOutput);
        cachedFfnInput->copyFrom(attentionOutput);

        // Feed-forward network
        std::shared_ptr<Matrix> mlpOut = feedForward->forward(attentionOutput);
        mlpOut->add(*attentionOutput); // Second residual connection

        return mlpOut;
    }

    std::shared_ptr<Matrix> EncoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        // ---- 1. Backprop through second residual connection and FFN ----
        // gradOutput flows to both the FFN and the residual connection
        auto gradFfnInputFromResidual = std::make_shared<Matrix>(*gradOutput);
        auto gradFfnInput = std::make_shared<Matrix>(*gradOutput);

        // Backprop through feedforward network
        auto gradFromFfn = feedForward->backward_with_targetloss(cachedFfnInput, gradFfnInput, learningRate);

        // Add gradient from residual connection
        gradFromFfn->add(*gradFfnInputFromResidual);

        // ---- 2. Backprop through first residual connection and self-attention ----
        // gradFromFfn flows to both the attention and the residual connection
        auto gradInputFromResidual = std::make_shared<Matrix>(*gradFromFfn);
        auto gradAttentionInput = std::make_shared<Matrix>(*gradFromFfn);

        // Backprop through self-attention (no context for encoder self-attention)
        auto [ gradFromAttention, gradContext ] = attention->backward(gradAttentionInput, cachedInput, nullptr);

        // Add gradient from residual connection
        gradFromAttention->add(*gradInputFromResidual);

        return gradFromAttention;
    }
} 
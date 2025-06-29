#include "DecoderBlock.h"
#include <memory>

namespace NNGL {
    DecoderBlock::DecoderBlock(int modelDim, int hiddenDim, int seqLen) {
        int headDim = modelDim; // same as modelDim for simplicity

        maskedSelfAttn = std::make_unique<AttentionBlock>(modelDim, headDim, seqLen, /*isMasked=*/true);
        crossAttn = std::make_unique<AttentionBlock>(modelDim, headDim, seqLen); // CrossAttention takes Q, K, V separately

        feedForward = std::make_unique<NeuralNetwork>(seqLen);
        feedForward->addLayer(headDim, hiddenDim, NNGL::ActivationFnType::RELU);
        feedForward->addLayer(hiddenDim, headDim, NNGL::ActivationFnType::RELU);

        // Initialize cache matrices
        cachedMaskedOut = std::make_shared<Matrix>(seqLen, modelDim);
        cachedCrossOut = std::make_shared<Matrix>(seqLen, modelDim);
        cachedDecoderInput = std::make_shared<Matrix>(seqLen, modelDim);
        cachedEncoderOutput = std::make_shared<Matrix>(seqLen, modelDim);
    }

    std::shared_ptr<Matrix> DecoderBlock::forward(
        std::shared_ptr<Matrix> decoderInput,
        std::shared_ptr<Matrix> encoderOutput
    ) {
        // Cache inputs for backprop
        cachedDecoderInput->copyFrom(decoderInput);
        cachedEncoderOutput->copyFrom(encoderOutput);

        auto maskedOut = maskedSelfAttn->forward(decoderInput);
        maskedOut->add(*decoderInput);  // first residual
        cachedMaskedOut->copyFrom(maskedOut);  // cache this intermediate result

        auto crossOut = crossAttn->forward(maskedOut, encoderOutput);
        crossOut->add(*maskedOut);      // second residual
        cachedCrossOut->copyFrom(crossOut);  // cache this intermediate result

        auto mlpOut = feedForward->forward(crossOut);
        mlpOut->add(*crossOut);         // third residual

        return mlpOut;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> gradOutput, float learningRate) {
        // ---- 1. Backprop through final residual connection and MLP ----
        auto gradCrossOutFromResidual = std::make_shared<Matrix>(*gradOutput);
        auto gradMlpInput = std::make_shared<Matrix>(*gradOutput);

        auto gradFromMlp = feedForward->backward_with_targetloss(cachedCrossOut, gradMlpInput, learningRate);
        gradFromMlp->add(*gradCrossOutFromResidual);

        // ---- 2. Backprop through second residual connection and Cross-Attention ----
        auto gradMaskedOutFromResidual = std::make_shared<Matrix>(*gradFromMlp);
        auto gradCrossInput = std::make_shared<Matrix>(*gradFromMlp);

        auto [gradFromCross, gradContext] = crossAttn->backward(gradCrossInput, cachedMaskedOut, cachedEncoderOutput);

        gradFromCross->add(*gradMaskedOutFromResidual);

        // ---- 3. Backprop through first residual connection and Masked Self-Attention ----
        auto gradDecoderInputFromResidual = std::make_shared<Matrix>(*gradFromCross);
        auto gradMaskedInput = std::make_shared<Matrix>(*gradFromCross);

        auto [gradFromMaskedSelf, maskedGradContext] = maskedSelfAttn->backward(gradMaskedInput, cachedDecoderInput, nullptr);
        gradFromMaskedSelf->add(*gradDecoderInputFromResidual);

        return gradFromMaskedSelf;
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> DecoderBlock::backwardWithEncoderGrad(
        std::shared_ptr<Matrix> gradOutput, float learningRate) {

        // ---- 1. Backprop through final residual connection and MLP ----
        auto gradCrossOutFromResidual = std::make_shared<Matrix>(*gradOutput);
        auto gradMlpInput = std::make_shared<Matrix>(*gradOutput);

        auto gradFromMlp = feedForward->backward_with_targetloss(cachedCrossOut, gradMlpInput, learningRate);
        gradFromMlp->add(*gradCrossOutFromResidual);

        // ---- 2. Backprop through second residual connection and Cross-Attention ----
        auto gradMaskedOutFromResidual = std::make_shared<Matrix>(*gradFromMlp);
        auto gradCrossInput = std::make_shared<Matrix>(*gradFromMlp);


        auto [gradFromCrossQuery, gradFromCrossEncoder] = crossAttn->backward(gradCrossInput, cachedMaskedOut, cachedEncoderOutput);

        gradFromCrossQuery->add(*gradMaskedOutFromResidual);

        // ---- 3. Backprop through first residual connection and Masked Self-Attention ----
        auto gradDecoderInputFromResidual = std::make_shared<Matrix>(*gradFromCrossQuery);
        auto gradMaskedInput = std::make_shared<Matrix>(*gradFromCrossQuery);

        auto [gradFromMaskedSelf, gradFromMaskedEncoder] = maskedSelfAttn->backward(gradMaskedInput, cachedDecoderInput, nullptr);
        gradFromMaskedSelf->add(*gradDecoderInputFromResidual);

        // Return BOTH gradients: decoder input gradient AND encoder output gradient
        return std::make_pair(gradFromMaskedSelf, gradFromCrossEncoder);
    }
} 
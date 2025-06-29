#pragma once

#include "AttentionBlock.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <memory>

namespace NNGL {
    class DecoderBlock {
    private:
        std::unique_ptr<AttentionBlock> maskedSelfAttn;   // Masked self-attention
        std::unique_ptr<AttentionBlock> crossAttn;       // Cross-attention (encoder-decoder)
        std::unique_ptr<NeuralNetwork> feedForward;

        std::shared_ptr<Matrix> cachedMaskedOut;
        std::shared_ptr<Matrix> cachedCrossOut;
        std::shared_ptr<Matrix> cachedDecoderInput;
        std::shared_ptr<Matrix> cachedEncoderOutput;

    public:
        DecoderBlock(int modelDim, int hiddenDim, int seqLen);
        
        std::shared_ptr<Matrix> forward(
            std::shared_ptr<Matrix> decoderInput,
            std::shared_ptr<Matrix> encoderOutput
        );
        
        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, float learningRate);
        
        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> backwardWithEncoderGrad(
            std::shared_ptr<Matrix> gradOutput, float learningRate);
    };
} 
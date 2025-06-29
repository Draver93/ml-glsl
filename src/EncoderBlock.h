#pragma once

#include "AttentionBlock.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <memory>

namespace NNGL {
    class EncoderBlock {
    private:
        std::unique_ptr<AttentionBlock> attention;
        std::unique_ptr<NeuralNetwork> feedForward;

        // Cache intermediate results for backpropagation
        std::shared_ptr<Matrix> cachedInput;
        std::shared_ptr<Matrix> cachedAttentionOutput;
        std::shared_ptr<Matrix> cachedFfnInput;

    public:
        EncoderBlock(int modelDim, int hiddenDim, int seqLen);
        
        std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> x);
        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, float learningRate);
    };
} 
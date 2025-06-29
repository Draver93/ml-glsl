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
        std::shared_ptr<Matrix> cached_input;
        std::shared_ptr<Matrix> cached_attention_output;
        std::shared_ptr<Matrix> cached_ffn_input;

    public:
        EncoderBlock(int model_dim, int hidden_dim, int seq_len);
        
        std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> x);
        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> grad_output, float learningRate);
    };
} 
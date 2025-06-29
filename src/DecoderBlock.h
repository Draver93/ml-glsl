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

        std::shared_ptr<Matrix> cached_masked_out;
        std::shared_ptr<Matrix> cached_cross_out;
        std::shared_ptr<Matrix> cached_decoder_input;
        std::shared_ptr<Matrix> cached_encoder_output;

    public:
        DecoderBlock(int model_dim, int hidden_dim, int seq_len);
        
        std::shared_ptr<Matrix> forward(
            std::shared_ptr<Matrix> decoder_input,
            std::shared_ptr<Matrix> encoder_output
        );
        
        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> grad_output, float learningRate);
        
        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> backwardWithEncoderGrad(
            std::shared_ptr<Matrix> grad_output, float learningRate);
    };
} 
#pragma once

#include "AttentionBlock.h"
#include "NeuralNetwork.h"

namespace NNGL {

    //x -> SelfAttention -> +residual -> FeedForward -> +residual -> out
    class EncoderBlock {
        std::unique_ptr<AttentionBlock> attention;
        std::unique_ptr<NeuralNetwork> feedForward;

    public:
        EncoderBlock(int model_dim, int hidden_dim, int seq_len) {
            int head_dim = model_dim; //for simplicity
            attention = std::make_unique<AttentionBlock>(model_dim, head_dim, seq_len, true);

            feedForward = std::make_unique<NeuralNetwork>(seq_len);
            feedForward->addLayer(head_dim, hidden_dim, NNGL::ActivationFnType::RELU);
            feedForward->addLayer(hidden_dim, head_dim, NNGL::ActivationFnType::RELU);
        }

        std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> x) {

            std::shared_ptr<Matrix> attentionOutput = attention->forward(x);
            attentionOutput->add((*x));  // first residual
            std::shared_ptr<Matrix> mlp_out = feedForward->forward(attentionOutput);
            mlp_out->add((*attentionOutput)); // second residual
            return mlp_out;
        }
    };

    class DecoderBlock {
        std::unique_ptr<AttentionBlock> maskedSelfAttn;   // Masked self-attention
        std::unique_ptr<AttentionBlock> crossAttn;       // Cross-attention (encoder-decoder)
        std::unique_ptr<NeuralNetwork> feedForward;

    public:
        DecoderBlock(int model_dim, int hidden_dim, int seq_len) {
            int head_dim = model_dim; // same as model_dim for simplicity

            maskedSelfAttn = std::make_unique<AttentionBlock>(model_dim, head_dim, seq_len, /*isMasked=*/true);
            crossAttn = std::make_unique<AttentionBlock>(model_dim, head_dim, seq_len); // CrossAttention takes Q, K, V separately

            feedForward = std::make_unique<NeuralNetwork>(seq_len);
            feedForward->addLayer(head_dim, hidden_dim, NNGL::ActivationFnType::RELU);
            feedForward->addLayer(hidden_dim, head_dim, NNGL::ActivationFnType::RELU);
        }

        std::shared_ptr<Matrix> forward(
            std::shared_ptr<Matrix> decoder_input,
            std::shared_ptr<Matrix> encoder_output
        ) {
            auto masked_out = maskedSelfAttn->forward(decoder_input);
            masked_out->add(*decoder_input);  // first residual

            auto cross_out = crossAttn->forward(masked_out, encoder_output);
            cross_out->add(*masked_out);      // second residual

            auto mlp_out = feedForward->forward(cross_out);
            mlp_out->add(*cross_out);         // third residual

            return mlp_out;
        }

    };

    class Transformer {
        std::unique_ptr<EncoderBlock> encoder;
        std::unique_ptr<DecoderBlock> decoder;
        std::unique_ptr<NeuralNetwork> outputProjection;  // W_out as NN layer

    public:
        Transformer(int model_dim, int hidden_dim, int seq_len, int vocab_size) {
            encoder = std::make_unique<EncoderBlock>(model_dim, hidden_dim, seq_len);
            decoder = std::make_unique<DecoderBlock>(model_dim, hidden_dim, seq_len);

            // Output projection: from model_dim to vocab_size
            outputProjection = std::make_unique<NeuralNetwork>(seq_len);
            outputProjection->addLayer(model_dim, vocab_size, NNGL::ActivationFnType::IDENTITY);
        }

        // Forward that takes encoder input tokens and returns next token idz
        int forward(std::shared_ptr<Matrix> encInputMat, std::shared_ptr<Matrix> decInputMat) {

            // 2. Encode input
            std::shared_ptr<Matrix> encOutputMat = encoder->forward(encInputMat);

            // 4. Decode
            std::shared_ptr<Matrix> decOutputMat = decoder->forward(decInputMat, encOutputMat);

            // 5. Project decoder output to vocab logits
            std::shared_ptr<Matrix> logits = outputProjection->forward(decOutputMat);

            int predicted_token = -1;
            float max_token = FLT_MIN;
            for (int i = 0; i < logits->cols; i++) 
                if (max_token < (*logits)(0, i)) {
                    max_token = (*logits)(0, i);
                    predicted_token = i;
                }
  
            return predicted_token;
        }
    };
}
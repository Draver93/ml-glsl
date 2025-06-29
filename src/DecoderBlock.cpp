#include "DecoderBlock.h"
#include <memory>

namespace NNGL {
    DecoderBlock::DecoderBlock(int model_dim, int hidden_dim, int seq_len) {
        int head_dim = model_dim; // same as model_dim for simplicity

        maskedSelfAttn = std::make_unique<AttentionBlock>(model_dim, head_dim, seq_len, /*isMasked=*/true);
        crossAttn = std::make_unique<AttentionBlock>(model_dim, head_dim, seq_len); // CrossAttention takes Q, K, V separately

        feedForward = std::make_unique<NeuralNetwork>(seq_len);
        feedForward->addLayer(head_dim, hidden_dim, NNGL::ActivationFnType::RELU);
        feedForward->addLayer(hidden_dim, head_dim, NNGL::ActivationFnType::RELU);

        // Initialize cache matrices
        cached_masked_out = std::make_shared<Matrix>(seq_len, model_dim);
        cached_cross_out = std::make_shared<Matrix>(seq_len, model_dim);
        cached_decoder_input = std::make_shared<Matrix>(seq_len, model_dim);
        cached_encoder_output = std::make_shared<Matrix>(seq_len, model_dim);
    }

    std::shared_ptr<Matrix> DecoderBlock::forward(
        std::shared_ptr<Matrix> decoder_input,
        std::shared_ptr<Matrix> encoder_output
    ) {
        // Cache inputs for backprop
        cached_decoder_input->copyFrom(decoder_input);
        cached_encoder_output->copyFrom(encoder_output);

        auto masked_out = maskedSelfAttn->forward(decoder_input);
        masked_out->add(*decoder_input);  // first residual
        cached_masked_out->copyFrom(masked_out);  // cache this intermediate result

        auto cross_out = crossAttn->forward(masked_out, encoder_output);
        cross_out->add(*masked_out);      // second residual
        cached_cross_out->copyFrom(cross_out);  // cache this intermediate result

        auto mlp_out = feedForward->forward(cross_out);
        mlp_out->add(*cross_out);         // third residual

        return mlp_out;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> grad_output, float learningRate) {
        // ---- 1. Backprop through final residual connection and MLP ----
        auto grad_cross_out_from_residual = std::make_shared<Matrix>(*grad_output);
        auto grad_mlp_input = std::make_shared<Matrix>(*grad_output);

        auto grad_from_mlp = feedForward->backward_with_targetloss(cached_cross_out, grad_mlp_input, learningRate);
        grad_from_mlp->add(*grad_cross_out_from_residual);

        // ---- 2. Backprop through second residual connection and Cross-Attention ----
        auto grad_masked_out_from_residual = std::make_shared<Matrix>(*grad_from_mlp);
        auto grad_cross_input = std::make_shared<Matrix>(*grad_from_mlp);

        auto [grad_from_cross, grad_context] = crossAttn->backward(grad_cross_input, cached_masked_out, cached_encoder_output);

        grad_from_cross->add(*grad_masked_out_from_residual);

        // ---- 3. Backprop through first residual connection and Masked Self-Attention ----
        auto grad_decoder_input_from_residual = std::make_shared<Matrix>(*grad_from_cross);
        auto grad_masked_input = std::make_shared<Matrix>(*grad_from_cross);

        auto [grad_from_masked_self, masked_grad_context] = maskedSelfAttn->backward(grad_masked_input, cached_decoder_input, nullptr);
        grad_from_masked_self->add(*grad_decoder_input_from_residual);

        return grad_from_masked_self;
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> DecoderBlock::backwardWithEncoderGrad(
        std::shared_ptr<Matrix> grad_output, float learningRate) {

        // ---- 1. Backprop through final residual connection and MLP ----
        auto grad_cross_out_from_residual = std::make_shared<Matrix>(*grad_output);
        auto grad_mlp_input = std::make_shared<Matrix>(*grad_output);

        auto grad_from_mlp = feedForward->backward_with_targetloss(cached_cross_out, grad_mlp_input, learningRate);
        grad_from_mlp->add(*grad_cross_out_from_residual);

        // ---- 2. Backprop through second residual connection and Cross-Attention ----
        auto grad_masked_out_from_residual = std::make_shared<Matrix>(*grad_from_mlp);
        auto grad_cross_input = std::make_shared<Matrix>(*grad_from_mlp);


        auto [grad_from_cross_query, grad_from_cross_encoder] = crossAttn->backward(grad_cross_input, cached_masked_out, cached_encoder_output);

        grad_from_cross_query->add(*grad_masked_out_from_residual);

        // ---- 3. Backprop through first residual connection and Masked Self-Attention ----
        auto grad_decoder_input_from_residual = std::make_shared<Matrix>(*grad_from_cross_query);
        auto grad_masked_input = std::make_shared<Matrix>(*grad_from_cross_query);

        auto [grad_from_masked_self, grad_from_masked_encoder] = maskedSelfAttn->backward(grad_masked_input, cached_decoder_input, nullptr);
        grad_from_masked_self->add(*grad_decoder_input_from_residual);

        // Return BOTH gradients: decoder input gradient AND encoder output gradient
        return std::make_pair(grad_from_masked_self, grad_from_cross_encoder);
    }
} 
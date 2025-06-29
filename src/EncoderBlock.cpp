#include "EncoderBlock.h"
#include <memory>

namespace NNGL {
    EncoderBlock::EncoderBlock(int model_dim, int hidden_dim, int seq_len) {
        int head_dim = model_dim; // for simplicity
        attention = std::make_unique<AttentionBlock>(model_dim, head_dim, seq_len, false); // No masking for encoder

        feedForward = std::make_unique<NeuralNetwork>(seq_len);
        feedForward->addLayer(head_dim, hidden_dim, NNGL::ActivationFnType::RELU);
        feedForward->addLayer(hidden_dim, head_dim, NNGL::ActivationFnType::RELU);

        // Initialize cache matrices
        cached_input = std::make_shared<Matrix>(seq_len, model_dim);
        cached_attention_output = std::make_shared<Matrix>(seq_len, model_dim);
        cached_ffn_input = std::make_shared<Matrix>(seq_len, model_dim);
    }

    std::shared_ptr<Matrix> EncoderBlock::forward(std::shared_ptr<Matrix> x) {
        // Cache input for backpropagation
        cached_input->copyFrom(x);

        // Self-attention
        std::shared_ptr<Matrix> attentionOutput = attention->forward(x);
        attentionOutput->add(*x);  // First residual connection

        // Cache attention output (after residual)
        cached_attention_output->copyFrom(attentionOutput);
        cached_ffn_input->copyFrom(attentionOutput);

        // Feed-forward network
        std::shared_ptr<Matrix> mlp_out = feedForward->forward(attentionOutput);
        mlp_out->add(*attentionOutput); // Second residual connection

        return mlp_out;
    }

    std::shared_ptr<Matrix> EncoderBlock::backward(std::shared_ptr<Matrix> grad_output, float learningRate) {
        // ---- 1. Backprop through second residual connection and FFN ----
        // grad_output flows to both the FFN and the residual connection
        auto grad_ffn_input_from_residual = std::make_shared<Matrix>(*grad_output);
        auto grad_ffn_input = std::make_shared<Matrix>(*grad_output);

        // Backprop through feedforward network
        auto grad_from_ffn = feedForward->backward_with_targetloss(cached_ffn_input, grad_ffn_input, learningRate);

        // Add gradient from residual connection
        grad_from_ffn->add(*grad_ffn_input_from_residual);

        // ---- 2. Backprop through first residual connection and self-attention ----
        // grad_from_ffn flows to both the attention and the residual connection
        auto grad_input_from_residual = std::make_shared<Matrix>(*grad_from_ffn);
        auto grad_attention_input = std::make_shared<Matrix>(*grad_from_ffn);

        // Backprop through self-attention (no context for encoder self-attention)
        auto [ grad_from_attention, grad_context ] = attention->backward(grad_attention_input, cached_input, nullptr);

        // Add gradient from residual connection
        grad_from_attention->add(*grad_input_from_residual);

        return grad_from_attention;
    }
} 
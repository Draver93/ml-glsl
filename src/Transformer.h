#pragma once

#include "SelfAttention.h"
#include "NeuralNetwork.h"

namespace NNGL {

    //x -> SelfAttention -> +residual -> FeedForward -> +residual -> out
    class EncoderBlock {
        std::unique_ptr<SelfAttention> attention;
        std::unique_ptr<NeuralNetwork> feedForward;

    public:
        EncoderBlock(int model_dim, int hidden_dim, int seq_len) {
            int head_dim = model_dim; //for simplicity
            attention = std::make_unique<SelfAttention>(model_dim, head_dim, seq_len, true);

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
        std::unique_ptr<SelfAttention> maskedSelfAttn;   // Masked self-attention
       // std::unique_ptr<CrossAttention> crossAttn;       // Cross-attention (encoder-decoder)
        std::unique_ptr<NeuralNetwork> feedForward;

    public:
        DecoderBlock(int model_dim, int hidden_dim, int seq_len) {
            int head_dim = model_dim; // same as model_dim for simplicity

            maskedSelfAttn = std::make_unique<SelfAttention>(model_dim, head_dim, seq_len, /*isMasked=*/true);
            //crossAttn = std::make_unique<CrossAttention>(model_dim, head_dim, seq_len); // CrossAttention takes Q, K, V separately

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
            return masked_out;

            //auto cross_out = crossAttn->forward(masked_out, encoder_output);
            //cross_out->add(*masked_out);      // second residual

            //auto mlp_out = feedForward->forward(cross_out);
            //mlp_out->add(*cross_out);         // third residual

            // return mlp_out;
        }
    };

	class Transformer {
	public:
		Transformer() {};
		~Transformer() {};
	};
}
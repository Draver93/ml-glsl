#pragma once

#include "SelfAttention.h"
#include "NeuralNetwork.h"

namespace NNGL {

    class EncoderBlock {
        std::unique_ptr<SelfAttention> attention;
        std::unique_ptr<NeuralNetwork> feedForward;
        // optional: LayerNorm norm1, norm2;

    public:
        EncoderBlock(int input_dim, int head_dim, int hidden_dim) {
        
        }

        std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> x) {
            /*auto attn = attention.forward(x);
            auto res1 = attn->add(*x);  // first residual

            auto mlp_out = feedForward.forward(res1);
            auto res2 = mlp_out->add(*res1);  // second residual

            return res2;*/
        }
    };

    class DecoderBlock {
        std::unique_ptr<SelfAttention> maskedSelfAttn;          // with causal mask
        std::unique_ptr<SelfAttention> crossAttn;               // Q: from decoder, K,V: from encoder
        std::unique_ptr<NeuralNetwork> feedForward;

    public:
        DecoderBlock(int model_dim, int head_dim, int hidden_dim) {
        }

        std::shared_ptr<Matrix> forward(
            std::shared_ptr<Matrix> decoder_input,
            std::shared_ptr<Matrix> encoder_output
        ) {
           /* auto masked_out = maskedSelfAttn.forward(decoder_input);
            auto res1 = masked_out->add(*decoder_input);

            auto cross_out = crossAttn.forward(res1, encoder_output);
            auto res2 = cross_out->add(*res1);

            auto mlp_out = feedForward.forward(res2);
            auto res3 = mlp_out->add(*res2);

            return res3;*/
        }
    };

	class Transformer {
	public:
		Transformer() {};
		~Transformer() {};
	};
}
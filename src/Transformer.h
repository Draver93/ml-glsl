#pragma once

#include "AttentionBlock.h"
#include "NeuralNetwork.h"
#include "BPE.h"

#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace NNGL {
    class EmbeddingBlock {
    private:
        std::mt19937 m_Generator;
        std::normal_distribution<float> m_Distribution;

        size_t m_VocabSize, m_ModelDim;
        std::unordered_map<std::string, std::vector<float>> m_Embeddings;

    public:
        EmbeddingBlock(size_t vocabSize, size_t modelDim) : 
            m_VocabSize(vocabSize), 
            m_ModelDim(modelDim),
            m_Generator(std::random_device{}()),
            m_Distribution(0.0f, 0.02f) {

            m_Embeddings.reserve(m_VocabSize);
        }

        std::shared_ptr<Matrix> forward(std::vector<std::string>& tokens) {
            std::vector<std::vector<float>> tmpVec; tmpVec.reserve(tokens.size());

            for (auto& t : tokens) {
                if (m_Embeddings.find(t) == m_Embeddings.end()) m_Embeddings[t] = initializeRandomVec();
                tmpVec.push_back(m_Embeddings[t]);
            }

            return std::make_shared<Matrix>(tmpVec);
        }

        std::vector<float> initializeRandomVec() {
            std::vector<float> vec(m_ModelDim);

            for (int i = 0; i < m_ModelDim; i++) { vec[i] = m_Distribution(m_Generator); }
            return vec;
        }

        void updateEmbedding(const std::string& token, const std::vector<float>& gradients, float learningRate) {
            auto it = m_Embeddings.find(token);
            if (it != m_Embeddings.end()) {
                for (size_t i = 0; i < m_ModelDim; i++) {
                    it->second[i] -= learningRate * gradients[i];
                }
            }
        }

        void save(const std::string& filename) const {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) throw std::runtime_error("Cannot open file for writing");

            // Write metadata
            size_t vocabSize = m_Embeddings.size();
            file.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));
            file.write(reinterpret_cast<const char*>(&m_ModelDim), sizeof(m_ModelDim));

            // Write embeddings
            for (const auto& [token, embedding] : m_Embeddings) {
                size_t tokenLength = token.length();
                file.write(reinterpret_cast<const char*>(&tokenLength), sizeof(tokenLength));
                file.write(token.c_str(), tokenLength);
                file.write(reinterpret_cast<const char*>(embedding.data()),
                    embedding.size() * sizeof(float));
            }
        }

        void load(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) throw std::runtime_error("Cannot open file for reading");

            // Read metadata
            size_t vocabSize;
            size_t modelDim;
            file.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
            file.read(reinterpret_cast<char*>(&modelDim), sizeof(modelDim));

            if (modelDim != m_ModelDim) {
                throw std::runtime_error("Model dimension mismatch");
            }

            m_Embeddings.clear();
            m_Embeddings.reserve(vocabSize);

            for (size_t i = 0; i < vocabSize; i++) {
                size_t tokenLength;
                file.read(reinterpret_cast<char*>(&tokenLength), sizeof(tokenLength));

                std::string token(tokenLength, '\0');
                file.read(&token[0], tokenLength);

                std::vector<float> embedding(m_ModelDim);
                file.read(reinterpret_cast<char*>(embedding.data()),
                    m_ModelDim * sizeof(float));

                m_Embeddings[token] = std::move(embedding);
            }
        }
    };

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
        std::shared_ptr<Matrix> backward( std::shared_ptr<Matrix> grad, std::shared_ptr<Matrix> input, float learningRate) {
            // ---- 1. Backprop through MLP ----
            auto grad_mlp_input = grad; // due to final residual: mlp_out + cross_out
            auto grad_from_mlp = feedForward->backward_with_targetloss(input, grad_mlp_input, learningRate);
            grad_from_mlp->add(*grad_mlp_input); // Add residual grad

            // ---- 2. Backprop through Cross-Attention ----
           /* auto grad_cross_input = grad_from_mlp;
            auto cross_attn_input = maskedSelfAttn->getOutput(); // Output from maskedSelfAttn
            auto grad_from_cross = crossAttn->backward(grad_cross_input, cross_attn_input, encoder_output); 
            grad_from_cross->add(*grad_cross_input); // Add residual grad

            // ---- 3. Backprop through Masked Self-Attention ----
            auto grad_masked_input = grad_from_cross;
            auto grad_from_self = maskedSelfAttn->backward(grad_masked_input, decoder_input);*/

            return nullptr;// grad_from_self;
        }
    };

    class Transformer {
    private:
        std::unique_ptr<BPE> m_Tokenizer;

        std::unique_ptr<EmbeddingBlock> m_Embedder;
        std::unique_ptr<EncoderBlock> m_Encoder;
        std::unique_ptr<DecoderBlock> m_Decoder;
        std::unique_ptr<NeuralNetwork> m_OutputProjection;  // W_out as NN layer

        size_t m_SeqLen, m_VocabSize;
    public:
        Transformer(std::string tokCheckpointFilepath, int modelDim, int hiddenDim, int seqLen) : m_SeqLen(seqLen) {

            m_Tokenizer = std::make_unique<BPE>();
            m_Tokenizer->load(tokCheckpointFilepath);
            m_VocabSize = m_Tokenizer->getVocabSize();


            m_Embedder = std::make_unique<EmbeddingBlock>(m_VocabSize, modelDim);
            m_Encoder = std::make_unique<EncoderBlock>(modelDim, hiddenDim, seqLen);
            m_Decoder = std::make_unique<DecoderBlock>(modelDim, hiddenDim, seqLen);

            // Output projection: from model_dim to vocab_size
            m_OutputProjection = std::make_unique<NeuralNetwork>(seqLen);
            m_OutputProjection->addLayer(modelDim, m_VocabSize, NNGL::ActivationFnType::IDENTITY);
        }

        void train(std::string& inputText) {


            // 1. Tokenize inputs
            std::vector<std::string> encInputTokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());

            // expected token
            std::string expected_tok = encInputTokens.back();
            encInputTokens.pop_back();

            while (encInputTokens.size() < m_SeqLen) encInputTokens.push_back("<PAD>");
            if (encInputTokens.size() > m_SeqLen) encInputTokens = std::vector<std::string>(encInputTokens.end() - m_SeqLen, encInputTokens.end());

            // 2. Prepare decoder inputs (shifted right) and targets
            std::vector<std::string> decInputTokens = { "<SOS>" };
            decInputTokens.insert(decInputTokens.end(), encInputTokens.begin(), encInputTokens.end() - 1);

            std::vector<std::string> targetTokens = encInputTokens; // For simplicity, let's assume autoencoder task
            while (targetTokens.size() < m_SeqLen) targetTokens.push_back("<PAD>");


            std::vector<float> expected_vec(m_VocabSize, 0.0f);
            train(encInputTokens, decInputTokens, expected_vec);
        }

        std::string eval(std::string& inputText) {
            std::vector<std::string> encInputTokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());
            while (encInputTokens.size() < m_SeqLen) encInputTokens.push_back("<PAD>");
            if (encInputTokens.size() > m_SeqLen) encInputTokens = std::vector<std::string>(encInputTokens.end() - m_SeqLen, encInputTokens.end());

            std::vector<std::string> decInputTokens(m_SeqLen, "<PAD>");
            decInputTokens.at(0) = "<SOS>";     // Start of generation

            int next_token_id = predictToken(forwardPass(encInputTokens, decInputTokens));
            return m_Tokenizer->getTokenById(next_token_id);
        }

    private:
        // Forward that takes encoder input tokens and returns next token idz
        std::shared_ptr<Matrix> forwardPass(std::vector<std::string>& encInputTokens, std::vector<std::string>& decInputTokens) {

            // 1. Embedd input
            std::shared_ptr<Matrix> encInputMat = m_Embedder->forward(encInputTokens);
            std::shared_ptr<Matrix> decInputMat = m_Embedder->forward(decInputTokens);

            // 2. Encode input
            std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat);

            // 4. Decode
            std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat);

            // 5. Project decoder output to vocab logits
            return m_OutputProjection->forward(decOutputMat);
        }
        
        void train(std::vector<std::string>& encInputTokens, std::vector<std::string>& decInputTokens, std::vector<float> expected) {
            float learningRate = 0.01f;
            // 1. Embedd input
            std::shared_ptr<Matrix> encInputMat = m_Embedder->forward(encInputTokens); //should add position enc
            std::shared_ptr<Matrix> decInputMat = m_Embedder->forward(decInputTokens); //should add position enc

            // 2. Encode input
            std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat);

            // 3. Decode
            std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat);

            // 4. FFN
            std::shared_ptr<Matrix> FFNGradMat = m_OutputProjection->backward(decOutputMat, std::make_shared<Matrix>(1, expected.size(), expected.data()), learningRate);
            
            // 5. Decode Grad
            std::shared_ptr<Matrix> decGradMat = m_Decoder->backward(FFNGradMat, decInputMat, learningRate);

            //Here we should 
            /*m_Embedder->updateEmbedding(decGradMat); //should remove position enc before applying grad

            // 6. Encode Grad
            std::shared_ptr<Matrix> encGradMat = m_Encoder->backward(decGradMat, learningRate);

            //Here we should 
            m_Embedder->updateEmbedding(encGradMat); //should remove position enc before applying grad*/
        }

        int predictToken(std::shared_ptr<Matrix> logits) {
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
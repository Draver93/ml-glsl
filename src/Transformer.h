#pragma once

#include "AttentionBlock.h"
#include "NeuralNetwork.h"

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

    };

    class Transformer {
    private:
        std::unique_ptr<EmbeddingBlock> m_Embedder;
        std::unique_ptr<EncoderBlock> m_Encoder;
        std::unique_ptr<DecoderBlock> m_Decoder;
        std::unique_ptr<NeuralNetwork> m_OutputProjection;  // W_out as NN layer

    public:
        Transformer(int modelDim, int hiddenDim, int seqLen, int vocabSize) {
            m_Embedder = std::make_unique<EmbeddingBlock>(vocabSize, modelDim);

            m_Encoder = std::make_unique<EncoderBlock>(modelDim, hiddenDim, seqLen);
            m_Decoder = std::make_unique<DecoderBlock>(modelDim, hiddenDim, seqLen);

            // Output projection: from model_dim to vocab_size
            m_OutputProjection = std::make_unique<NeuralNetwork>(seqLen);
            m_OutputProjection->addLayer(modelDim, vocabSize, NNGL::ActivationFnType::IDENTITY);
        }

        // Forward that takes encoder input tokens and returns next token idz
        int forward(std::vector<std::string> &encInputTokens, std::vector<std::string>& decInputTokens) {

            // 1. Embedd input
            std::shared_ptr<Matrix> encInputMat = m_Embedder->forward(encInputTokens);
            std::shared_ptr<Matrix> decInputMat = m_Embedder->forward(decInputTokens);

            // 2. Encode input
            std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat);

            // 4. Decode
            std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat);

            // 5. Project decoder output to vocab logits
            std::shared_ptr<Matrix> logits = m_OutputProjection->forward(decOutputMat);

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
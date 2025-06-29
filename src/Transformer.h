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

        size_t m_VocabSize, m_ModelDim, m_MaxSeqLen;
        std::unordered_map<std::string, std::vector<float>> m_Embeddings;

        std::shared_ptr<Matrix> m_PositionalEncodingMat;

        std::shared_ptr<Shader> m_ApplyPosEncodingCompute;
        std::shared_ptr<Shader> m_RemovePosEncodingCompute;

    public:
        EmbeddingBlock(size_t vocabSize, size_t modelDim, size_t maxSeqLen = 512) :
            m_VocabSize(vocabSize), 
            m_ModelDim(modelDim),
            m_MaxSeqLen(maxSeqLen),
            m_Generator(std::random_device{}()),
            m_Distribution(0.0f, 0.02f) {

            m_Embeddings.reserve(m_VocabSize);

            // Initialize positional encoding matrix
            initializePositionalEncoding();

            // Load compute shaders for positional encoding
            m_ApplyPosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/apply_pos_encoding.comp");
            m_RemovePosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/remove_pos_encoding.comp");
        }

        std::vector<float> initializeRandomVec() {
            std::vector<float> vec(m_ModelDim);

            for (int i = 0; i < m_ModelDim; i++) { vec[i] = m_Distribution(m_Generator); }
            return vec;
        }

        std::shared_ptr<Matrix> forward(std::vector<std::string>& tokens) {
            std::vector<std::vector<float>> tmpVec; tmpVec.reserve(tokens.size());

            for (auto& t : tokens) {
                if (m_Embeddings.find(t) == m_Embeddings.end()) m_Embeddings[t] = initializeRandomVec();
                tmpVec.push_back(m_Embeddings[t]);
            }
            return std::make_shared<Matrix>(tmpVec);
        }

        std::shared_ptr<Matrix> backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate) {
            if (!gradOutput || gradOutput->cols != m_ModelDim) {
                throw std::runtime_error("Invalid gradient dimensions");
            }

            // Download gradients from GPU
            gradOutput->downloadFromGPU();

            // Update embeddings using cached tokens
            size_t minSize = std::min(tokens.size(), static_cast<size_t>(gradOutput->rows));

            for (size_t i = 0; i < minSize; ++i) {
                const std::string& token = tokens[i];
                auto it = m_Embeddings.find(token);

                if (it != m_Embeddings.end()) {
                    for (size_t j = 0; j < m_ModelDim; ++j) {
                        it->second[j] -= learningRate * (*gradOutput)(i, j);
                    }
                }
            }

            // Return the gradient (no further backprop beyond embeddings)
            return nullptr;
        }

        void applyPositionalEncoding(std::shared_ptr<Matrix> embeddings) {
            if (!embeddings || embeddings->cols != m_ModelDim) {
                throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding");
            }

            size_t seqLen = std::min(static_cast<size_t>(embeddings->rows), m_MaxSeqLen);

            // Upload matrices to GPU
            embeddings->uploadToGPU();
            m_PositionalEncodingMat->uploadToGPU();

            // Bind buffers
            m_ApplyPosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", embeddings->buffer);
            m_ApplyPosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", m_PositionalEncodingMat->buffer);

            // Set uniforms
            m_ApplyPosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
            m_ApplyPosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));

            // Dispatch compute shader
            int workgroups_x = (seqLen + 15) / 16;
            int workgroups_y = (m_ModelDim + 15) / 16;
            m_ApplyPosEncodingCompute->dispatch(workgroups_x, workgroups_y, 1);

            embeddings->downloadFromGPU();

            // Unbind buffers
            for (int i = 0; i <= 1; ++i) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            }

        }

        void removePositionalEncoding(std::shared_ptr<Matrix> embeddings) {
            if (!embeddings || embeddings->cols != m_ModelDim) {
                throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding removal");
            }

            size_t seqLen = std::min(static_cast<size_t>(embeddings->rows), m_MaxSeqLen);

            // Upload matrices to GPU
            embeddings->uploadToGPU();
            m_PositionalEncodingMat->uploadToGPU();

            // Bind buffers
            m_RemovePosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", embeddings->buffer);
            m_RemovePosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", m_PositionalEncodingMat->buffer);

            // Set uniforms
            m_RemovePosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
            m_RemovePosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));

            // Dispatch compute shader
            int workgroups_x = (seqLen + 15) / 16;
            int workgroups_y = (m_ModelDim + 15) / 16;
            m_RemovePosEncodingCompute->dispatch(workgroups_x, workgroups_y, 1);

            embeddings->downloadFromGPU();

            // Unbind buffers
            for (int i = 0; i <= 1; ++i) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            }
        }

        void save(const std::string& filename) const {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) throw std::runtime_error("Cannot open file for writing");

            // Write metadata
            size_t vocabSize = m_Embeddings.size();
            file.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));
            file.write(reinterpret_cast<const char*>(&m_ModelDim), sizeof(m_ModelDim));
            file.write(reinterpret_cast<const char*>(&m_MaxSeqLen), sizeof(m_MaxSeqLen));

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
            size_t maxSeqLen;
            file.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
            file.read(reinterpret_cast<char*>(&modelDim), sizeof(modelDim));
            file.read(reinterpret_cast<char*>(&maxSeqLen), sizeof(maxSeqLen));

            if (modelDim != m_ModelDim) {
                throw std::runtime_error("Model dimension mismatch");
            }

            if (maxSeqLen != m_MaxSeqLen) {
                m_MaxSeqLen = maxSeqLen;
                // Reinitialize positional encoding with new sequence length
                initializePositionalEncoding();
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
   
 private:
        void initializePositionalEncoding() {
            // Create positional encoding matrix [max_seq_len, model_dim]
            m_PositionalEncodingMat = std::make_shared<Matrix>(m_MaxSeqLen, m_ModelDim);

            // Initialize with sinusoidal positional encoding
            for (size_t pos = 0; pos < m_MaxSeqLen; ++pos) {
                for (size_t i = 0; i < m_ModelDim; ++i) {
                    float angle = pos / std::pow(10000.0f, 2.0f * (i / 2.0f) / m_ModelDim);
                    if (i % 2 == 0) {
                        (*m_PositionalEncodingMat)(pos, i) = std::sin(angle);
                    }
                    else {
                        (*m_PositionalEncodingMat)(pos, i) = std::cos(angle);
                    }
                }
            }

            // Upload to GPU
            m_PositionalEncodingMat->uploadToGPU();
        }
    };

    class EncoderBlock {
        std::unique_ptr<AttentionBlock> attention;
        std::unique_ptr<NeuralNetwork> feedForward;

        // Cache intermediate results for backpropagation
        std::shared_ptr<Matrix> cached_input;
        std::shared_ptr<Matrix> cached_attention_output;
        std::shared_ptr<Matrix> cached_ffn_input;

    public:
        EncoderBlock(int model_dim, int hidden_dim, int seq_len) {
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

        std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> x) {
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

        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> grad_output, float learningRate) {
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
    };

    class DecoderBlock {
        std::unique_ptr<AttentionBlock> maskedSelfAttn;   // Masked self-attention
        std::unique_ptr<AttentionBlock> crossAttn;       // Cross-attention (encoder-decoder)
        std::unique_ptr<NeuralNetwork> feedForward;

        std::shared_ptr<Matrix> cached_masked_out;
        std::shared_ptr<Matrix> cached_cross_out;
        std::shared_ptr<Matrix> cached_decoder_input;
        std::shared_ptr<Matrix> cached_encoder_output;

    public:
        DecoderBlock(int model_dim, int hidden_dim, int seq_len) {
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

        std::shared_ptr<Matrix> forward(
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
        std::shared_ptr<Matrix> backward( std::shared_ptr<Matrix> grad_output, float learningRate ) {
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
    
        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> backwardWithEncoderGrad(
            std::shared_ptr<Matrix> grad_output, float learningRate) {

            // ---- 1. Backprop through final residual connection and MLP ----
            auto grad_cross_out_from_residual = std::make_shared<Matrix>(*grad_output);
            auto grad_mlp_input = std::make_shared<Matrix>(*grad_output);

            auto grad_from_mlp = feedForward->backward_with_targetloss(cached_cross_out, grad_mlp_input, learningRate);
            grad_from_mlp->add(*grad_cross_out_from_residual);

            // ---- 2. Backprop through second residual connection and Cross-Attention ----
            auto grad_masked_out_from_residual = std::make_shared<Matrix>(*grad_from_mlp);
            auto grad_cross_input = std::make_shared<Matrix>(*grad_from_mlp);

            // CRITICAL FIX: Cross-attention backward should return gradients for BOTH inputs
            // Query input (from decoder) AND Key/Value input (from encoder)
            // [Gradient for decoder input (query), Gradient for encoder output (key/value)]
            auto [grad_from_cross_query, grad_from_cross_encoder] = crossAttn->backward(grad_cross_input, cached_masked_out, cached_encoder_output);

            grad_from_cross_query->add(*grad_masked_out_from_residual);

            // ---- 3. Backprop through first residual connection and Masked Self-Attention ----
            auto grad_decoder_input_from_residual = std::make_shared<Matrix>(*grad_from_cross_query);
            auto grad_masked_input = std::make_shared<Matrix>(*grad_from_cross_query);

            auto [grad_from_masked_self, grad_from_masked_encoder] = maskedSelfAttn->backward(grad_masked_input, cached_decoder_input, nullptr);
            grad_from_masked_self->add(*grad_decoder_input_from_residual);

            // Return BOTH gradients: decoder input gradient AND encoder output gradienta
            return std::make_pair(grad_from_masked_self, grad_from_cross_encoder);
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
        size_t m_trainStep = 0;
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

        void train(const std::string & inputText) {
            // Tokenize the entire input
            std::vector<std::string> tokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());

            if (tokens.size() < 2) {
                //std::cerr << "Warning: Input too short for training" << std::endl;
                return;
            }

            // Train on sliding windows of the sequence
            trainOnSequence(tokens);
        }

        // Training on a sliding window of tokens from a long sequence
        void trainOnSequence(const std::vector<std::string>& longSequence, size_t windowSize = 0, float learningRate = 0.001f) {
            if (windowSize == 0) windowSize = m_SeqLen + 1; // +1 because we need input + target

            if (longSequence.size() < windowSize) {
                trainNextToken(longSequence, learningRate);
                return;
            }

            // Sliding window approach
            for (size_t i = 0; i <= longSequence.size() - windowSize; ++i) {
                std::vector<std::string> window(
                    longSequence.begin() + i,
                    longSequence.begin() + i + windowSize
                );
                trainNextToken(window, learningRate);
            }
        }

        void trainNextToken(const std::vector<std::string>& inputTokens, float learningRate = 0.001f) {
            if (inputTokens.size() < 2) {
                throw std::runtime_error("Need at least 2 tokens for next-token prediction");
            }

            // Prepare input sequence (all tokens except the last one)
            std::vector<std::string> contextTokens(inputTokens.begin(), inputTokens.end() - 1);

            // Target is the last token (what we want to predict)
            std::string targetToken = inputTokens.back();
            size_t targetTokenId = m_Tokenizer->getTokenByName(targetToken);

            // Pad or truncate context to sequence length
            while (contextTokens.size() < m_SeqLen) {
                contextTokens.push_back("<PAD>");
            }
            if (contextTokens.size() > m_SeqLen) {
                contextTokens = std::vector<std::string>(contextTokens.end() - m_SeqLen, contextTokens.end());
            }

            // For decoder-only architecture (like GPT), we use the same tokens for encoder and decoder
            // but shift decoder input by one position
            std::vector<std::string> decoderTokens = contextTokens;

            // Create one-hot target vector
            std::vector<float> targetOneHot(m_VocabSize, 0.0f);
            if (targetTokenId >= 0 && targetTokenId < m_VocabSize) {
                targetOneHot[targetTokenId] = 1.0f;
            }

            // Forward pass
            std::shared_ptr<Matrix> logits = forwardPass(contextTokens, decoderTokens);

            // Compute loss and gradients
            std::shared_ptr<Matrix> targetMat = std::make_shared<Matrix>(1, m_VocabSize, targetOneHot.data());

            // Backward pass
            backwardPass(contextTokens, decoderTokens, targetMat, learningRate);
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
            m_Embedder->applyPositionalEncoding(encInputMat);

            std::shared_ptr<Matrix> decInputMat = m_Embedder->forward(decInputTokens);
            m_Embedder->applyPositionalEncoding(decInputMat);

            // 2. Encode input
            std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat);

            // 4. Decode
            std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat);

            // 5. Project decoder output to vocab logits
            return m_OutputProjection->forward(decOutputMat);
        }
        
        void backwardPass(const std::vector<std::string>& encInputTokens,
            const std::vector<std::string>& decInputTokens,
            std::shared_ptr<Matrix> targetMat,
            float learningRate) {

            // 1. Embed inputs (forward pass needed for caching)
            std::shared_ptr<Matrix> encInputMat = m_Embedder->forward(const_cast<std::vector<std::string>&>(encInputTokens));
            m_Embedder->applyPositionalEncoding(encInputMat);

            std::shared_ptr<Matrix> decInputMat = m_Embedder->forward(const_cast<std::vector<std::string>&>(decInputTokens));
            m_Embedder->applyPositionalEncoding(decInputMat);

            // 2. Forward through encoder
            std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat);

            // 3. Forward through decoder
            std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat);

            // 4. Backward through output projection
            std::shared_ptr<Matrix> outputGrad = m_OutputProjection->backward(decOutputMat, targetMat, learningRate);
            printGradientHeatmap(outputGrad);
            // 5. Backward through decoder - THIS IS THE KEY FIX
            // The decoder backward should return TWO gradients:
            // - Gradient w.r.t. decoder input
            // - Gradient w.r.t. encoder output (from cross-attention)
            std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> decoderGrads =
                m_Decoder->backwardWithEncoderGrad(outputGrad, learningRate);

            std::shared_ptr<Matrix> decGrad = decoderGrads.first;      // Gradient for decoder input
            std::shared_ptr<Matrix> encOutputGrad = decoderGrads.second; // Gradient for encoder output

            // Update decoder embeddings
            m_Embedder->removePositionalEncoding(decGrad);
            m_Embedder->backward(decInputTokens, decGrad, learningRate);

            // 6. Backward through encoder using the gradient from decoder's cross-attention
            // THIS IS THE CORRECTED GRADIENT FLOW
            std::shared_ptr<Matrix> encGrad = m_Encoder->backward(encOutputGrad, learningRate);

            // Update encoder embeddings
            m_Embedder->removePositionalEncoding(encGrad);
            m_Embedder->backward(encInputTokens, encGrad, learningRate);
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
    
        void printGradientHeatmap(std::shared_ptr<Matrix> mat) {
            const std::string colors[] = {
                "\033[48;5;17m", "\033[48;5;18m", "\033[48;5;19m", "\033[48;5;20m", "\033[48;5;21m",
                "\033[48;5;38m", "\033[48;5;44m", "\033[48;5;51m", "\033[48;5;87m", "\033[48;5;123m",
                "\033[48;5;159m", "\033[48;5;190m", "\033[48;5;226m", "\033[48;5;220m", "\033[48;5;202m",
                "\033[48;5;196m", "\033[0m"
            };
            constexpr int COLOR_COUNT = sizeof(colors) / sizeof(colors[0]) - 1;

            int rows = mat->rows;
            int cols = mat->cols;
            const float* data = mat->flatVec.data();

            const int displaySize = 20;
            int rowStep = std::max(rows / displaySize, 1);
            int colStep = std::max(cols / displaySize, 1);

            std::vector<float> sampledValues;
            for (int i = 0; i < rows; i += rowStep) {
                for (int j = 0; j < cols; j += colStep) {
                    sampledValues.push_back(data[i * cols + j]);
                }
            }

            // Normalize sampled values
            auto [minIt, maxIt] = std::minmax_element(sampledValues.begin(), sampledValues.end());
            float minVal = *minIt, maxVal = *maxIt;
            float range = (maxVal - minVal) > 1e-9f ? (maxVal - minVal) : 1.0f;

            // Render heatmap
            size_t idx = 0;
            for (int i = 0; i < displaySize && i * rowStep < rows; ++i) {
                for (int j = 0; j < displaySize && j * colStep < cols; ++j) {
                    float normalized = (sampledValues[idx++] - minVal) / range;
                    int colorIdx = static_cast<int>(normalized * (COLOR_COUNT - 1));
                    std::cout << colors[colorIdx] << "  " << colors[COLOR_COUNT];  // Reset color
                }
                std::cout << "\n";
            }
            std::cout << colors[COLOR_COUNT] << std::endl;
        }
    };
}
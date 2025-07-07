#include "Transformer.h"
#include "Logger.h"

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cfloat>
#include <sstream>

namespace NNGL {
    Transformer::Transformer(std::string tokCheckpointFilepath, int modelDim, int hiddenDim, int seqLen) : m_SeqLen(seqLen) {
        m_Tokenizer = std::make_unique<BPE>();
        m_Tokenizer->load(tokCheckpointFilepath);
        m_Tokenizer->addToken("<PAD>");
        m_Tokenizer->addToken("<SOS>");
        m_Tokenizer->addToken("<EOS>");

        m_VocabSize = m_Tokenizer->getVocabSize();

        m_Embedder = std::make_unique<EmbeddingBlock>(m_VocabSize, modelDim);
        m_Encoder = std::make_unique<EncoderBlock>(modelDim, hiddenDim, seqLen);
        m_Decoder = std::make_unique<DecoderBlock>(modelDim, hiddenDim, seqLen);

        // Output projection: from model_dim to vocab_size
        m_OutputProjection = std::make_unique<NeuralNetwork>(seqLen);
        m_OutputProjection->addLayer(modelDim, m_VocabSize, NNGL::ActivationFnType::IDENTITY);
    }

    void Transformer::train(const std::string& inputText) {
        // Tokenize the entire input
        std::vector<std::string> tokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());

        if (tokens.size() < 2) {
            return;
        }

        // Convert to integer tokens for optimized training
        std::vector<int> tokenIds = stringToTokenIds(tokens);
        trainOnSequenceInt(tokenIds);
    }

    void Transformer::trainOnSequence(const std::vector<std::string>& longSequence, size_t windowSize, float learningRate) {
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

    void Transformer::trainOnSequenceInt(const std::vector<int>& longSequence, size_t windowSize, float learningRate) {
        if (windowSize == 0) windowSize = m_SeqLen + 1;

        if (longSequence.size() < windowSize) {
            trainNextTokenInt(longSequence, learningRate);
            return;
        }

        // Optimized sliding window - reuse vectors
        std::vector<int> window(windowSize);
        for (size_t i = 0; i <= longSequence.size() - windowSize; ++i) {
            // Copy window data without creating new vector
            std::copy(longSequence.begin() + i, longSequence.begin() + i + windowSize, window.begin());
            trainNextTokenInt(window, learningRate);
        }
    }

    void Transformer::trainNextToken(const std::vector<std::string>& inputTokens, float learningRate) {
        if (inputTokens.size() < 2) {
            throw std::runtime_error("Need at least 2 tokens for next-token prediction");
        }

        // Convert to integer tokens and use optimized version
        std::vector<int> tokenIds = stringToTokenIds(inputTokens);
        trainNextTokenInt(tokenIds, learningRate);
    }

    void Transformer::trainNextTokenInt(const std::vector<int>& inputTokens, float learningRate) {
        if (inputTokens.size() < 2) {
            throw std::runtime_error("Need at least 2 tokens for next-token prediction");
        }

        // Prepare input sequence (all tokens except the last one)
        std::vector<int> contextTokens(inputTokens.begin(), inputTokens.end() - 1);

        // Target is the last token (what we want to predict)
        int targetTokenId = inputTokens.back();

        // Pad or truncate context to sequence length
        while (contextTokens.size() < m_SeqLen) {
            contextTokens.push_back(0); // PAD token ID
        }
        if (contextTokens.size() > m_SeqLen) {
            contextTokens = std::vector<int>(contextTokens.end() - m_SeqLen, contextTokens.end());
        }

        // For decoder-only architecture, use the same tokens for encoder and decoder
        std::vector<int> decoderTokens = contextTokens;

        // Forward pass
        std::shared_ptr<Matrix> logits = forwardPassInt(contextTokens, decoderTokens);

        // Backward pass with integer target
        backwardPassInt(contextTokens, decoderTokens, targetTokenId, learningRate);
    }

    std::string Transformer::eval(std::string& inputText) {
        std::vector<std::string> encInputTokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());
        while (encInputTokens.size() < m_SeqLen) encInputTokens.push_back("<PAD>");
        if (encInputTokens.size() > m_SeqLen) encInputTokens = std::vector<std::string>(encInputTokens.end() - m_SeqLen, encInputTokens.end());

        std::vector<std::string> decInputTokens(m_SeqLen, "<PAD>");
        decInputTokens.at(0) = "<SOS>";     // Start of generation
        auto result = forwardPass(encInputTokens, decInputTokens);
        int nextTokenId = predictToken(result);
        return m_Tokenizer->getTokenById(nextTokenId);
    }

    std::shared_ptr<Matrix> Transformer::forwardPass(std::vector<std::string>& encInputTokens, std::vector<std::string>& decInputTokens) {
        // Convert to integer tokens and use optimized version
        std::vector<int> encTokenIds = stringToTokenIds(encInputTokens);
        std::vector<int> decTokenIds = stringToTokenIds(decInputTokens);
        return forwardPassInt(encTokenIds, decTokenIds);
    }

    std::shared_ptr<Matrix> Transformer::forwardPassInt(const std::vector<int>& encInputTokens, const std::vector<int>& decInputTokens) {
        // 1. Get cached embeddings or create new ones
        std::shared_ptr<Matrix> encInputMat = getCachedEmbedding(encInputTokens);
        m_Embedder->applyPositionalEncoding(encInputMat);

        std::shared_ptr<Matrix> decInputMat = getCachedEmbedding(decInputTokens);
        m_Embedder->applyPositionalEncoding(decInputMat);

        // 2. Encode input
        std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat);

        // 3. Decode
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat);

        // 4. Project decoder output to vocab logits
        return m_OutputProjection->forward(decOutputMat);
    }

    void Transformer::backwardPass(const std::vector<std::string>& encInputTokens,
        const std::vector<std::string>& decInputTokens,
        std::shared_ptr<Matrix> targetMat,
        float learningRate) {

        // Convert to integer tokens and use optimized version
        std::vector<int> encTokenIds = stringToTokenIds(encInputTokens);
        std::vector<int> decTokenIds = stringToTokenIds(decInputTokens);

        // Extract target token ID from one-hot vector
        int targetTokenId = -1;
        for (int i = 0; i < targetMat->cols; i++) {
            if ((*targetMat)(0, i) > 0.5f) {
                targetTokenId = i;
                break;
            }
        }

        backwardPassInt(encTokenIds, decTokenIds, targetTokenId, learningRate);
    }

    void Transformer::backwardPassInt(const std::vector<int>& encInputTokens,
        const std::vector<int>& decInputTokens,
        int targetTokenId,
        float learningRate) {

        // 1. Get cached embeddings
        std::shared_ptr<Matrix> encInputMat = getCachedEmbedding(encInputTokens);
        m_Embedder->applyPositionalEncoding(encInputMat);

        std::shared_ptr<Matrix> decInputMat = getCachedEmbedding(decInputTokens);
        m_Embedder->applyPositionalEncoding(decInputMat);

        // 2. Forward through encoder
        std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat);

        // 3. Forward through decoder
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat);

        // 4. Create sparse target (much more efficient than one-hot)
        std::vector<float> targetSparse(m_VocabSize, 0.0f);
        if (targetTokenId >= 0 && targetTokenId < m_VocabSize) {
            targetSparse[targetTokenId] = 1.0f;
        }
        std::shared_ptr<Matrix> targetMat = std::make_shared<Matrix>(1, m_VocabSize, targetSparse.data());

        // 5. Backward through output projection
        std::shared_ptr<Matrix> outputGrad = m_OutputProjection->backward(decOutputMat, targetMat, learningRate);

        // Apply gradient clipping to prevent explosion
        float maxGradNorm = 1.0f; // Clip gradients to norm of 1.0
        float gradNorm = 0.0f;

        // Check for NaN or infinite values and clamp them
        for (int i = 0; i < outputGrad->rows * outputGrad->cols; ++i) {
            if (std::isnan(outputGrad->flatVec[i]) || std::isinf(outputGrad->flatVec[i])) {
                outputGrad->flatVec[i] = 0.0f;
            }
            gradNorm += outputGrad->flatVec[i] * outputGrad->flatVec[i];
        }
        gradNorm = std::sqrt(gradNorm);

        if (gradNorm > maxGradNorm) {
            float scale = maxGradNorm / gradNorm;
            LOG_TRACE("Gradient clipping applied: norm=" + std::to_string(gradNorm) + ", scale=" + std::to_string(scale));
            for (int i = 0; i < outputGrad->rows * outputGrad->cols; ++i) {
                outputGrad->flatVec[i] *= scale;
            }
        }

        printGradientHeatmap(outputGrad);

        // 6. Backward through decoder
        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> decoderGrads =
            m_Decoder->backwardWithEncoderGrad(outputGrad, learningRate);

        std::shared_ptr<Matrix> decGrad = decoderGrads.first;
        std::shared_ptr<Matrix> encOutputGrad = decoderGrads.second;

        // Update decoder embeddings
        m_Embedder->removePositionalEncoding(decGrad);
        // Note: We need to convert back to strings for embedding backward pass
        std::vector<std::string> decInputStrings = tokenIdsToStrings(decInputTokens);
        m_Embedder->backward(decInputStrings, decGrad, learningRate);

        // 7. Backward through encoder
        std::shared_ptr<Matrix> encGrad = m_Encoder->backward(encOutputGrad, learningRate);

        // Update encoder embeddings
        m_Embedder->removePositionalEncoding(encGrad);
        std::vector<std::string> encInputStrings = tokenIdsToStrings(encInputTokens);
        m_Embedder->backward(encInputStrings, encGrad, learningRate);
    }

    int Transformer::predictToken(std::shared_ptr<Matrix> probabilities) {
        int predictedToken = -1;
        float maxProb = (*probabilities)(0, 0);
        for (int i = 0; i < probabilities->cols; i++) {
            if (maxProb < (*probabilities)(0, i)) {
                maxProb = (*probabilities)(0, i);
                predictedToken = i;
            }
        }
        return predictedToken;
    }

    // Token conversion helpers
    std::vector<int> Transformer::stringToTokenIds(const std::vector<std::string>& tokens) {
        std::vector<int> tokenIds;
        tokenIds.reserve(tokens.size());

        for (const auto& token : tokens) {
            size_t tokenId = m_Tokenizer->getTokenByName(token);
            if (tokenId >= 0 && tokenId < m_VocabSize) {
                tokenIds.push_back(static_cast<int>(tokenId));
            }
            else {
                tokenIds.push_back(0); // Unknown token -> PAD
            }
        }

        return tokenIds;
    }

    std::vector<std::string> Transformer::tokenIdsToStrings(const std::vector<int>& tokenIds) {
        std::vector<std::string> tokens;
        tokens.reserve(tokenIds.size());

        for (int tokenId : tokenIds) {
            if (tokenId >= 0 && tokenId < m_VocabSize) {
                tokens.push_back(m_Tokenizer->getTokenById(tokenId));
            }
            else {
                tokens.push_back("<PAD>");
            }
        }

        return tokens;
    }

    // Embedding caching
    std::shared_ptr<Matrix> Transformer::getCachedEmbedding(const std::vector<int>& tokens) {
        // Create cache key from token IDs
        std::stringstream ss;
        for (int tokenId : tokens) {
            ss << tokenId << ",";
        }
        std::string cacheKey = ss.str();

        // Check cache first
        {
            std::lock_guard<std::mutex> lock(m_CacheMutex);
            auto it = m_EmbeddingCache.find(cacheKey);
            if (it != m_EmbeddingCache.end()) {
                return it->second;
            }
        }

        // Convert to strings and create embedding
        std::vector<std::string> tokenStrings = tokenIdsToStrings(tokens);
        std::shared_ptr<Matrix> embedding = m_Embedder->forward(tokenStrings);

        // Cache the result
        {
            std::lock_guard<std::mutex> lock(m_CacheMutex);
            // Limit cache size to prevent memory explosion
            if (m_EmbeddingCache.size() > 1000) {
                m_EmbeddingCache.clear();
            }
            m_EmbeddingCache[cacheKey] = embedding;
        }

        return embedding;
    }

    void Transformer::printGradientHeatmap(std::shared_ptr<Matrix> mat) {
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
}
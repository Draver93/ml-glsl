#include "Transformer.h"
#include "Logger.h"

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cfloat>
#include <sstream>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision

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
        m_OutputProjection = std::make_unique<NeuralNetwork>(1);  // FIXED: Use batch size 1
        m_OutputProjection->addLayer(modelDim, m_VocabSize, NNGL::ActivationFnType::IDENTITY);
        
        // Initialize training statistics
        m_TrainingSteps = 0;
        m_CurrentLoss = 0.0f;
    }

    void Transformer::train(const std::string& inputText, float learningRate) {
        // Tokenize the entire input
        std::vector<std::string> tokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());

        if (tokens.size() < 2) {
            return;
        }

        // Add EOS token to the end for proper training
        tokens.push_back("<EOS>");
        trainOnSequence(tokens, 0, learningRate);
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

    void Transformer::trainNextToken(const std::vector<std::string>& inputTokens, float learningRate) {
        if (inputTokens.size() < 2) {
            throw std::runtime_error("Need at least 2 tokens for next-token prediction");
        }
        // Prepare input sequence (all tokens except the last one)
        std::vector<std::string> contextTokens(inputTokens.begin(), inputTokens.end() - 1);
        // Target is the last token (what we want to predict)
        std::string targetToken = inputTokens.back();
        // Pad or truncate context to sequence length
        while (contextTokens.size() < m_SeqLen) {
            contextTokens.push_back("<PAD>");
        }
        if (contextTokens.size() > m_SeqLen) {
            contextTokens = std::vector<std::string>(contextTokens.end() - m_SeqLen, contextTokens.end());
        }
        // For decoder-only architecture, use the same tokens for encoder and decoder
        std::vector<std::string> decoderTokens = contextTokens;
        // Forward pass
        std::shared_ptr<Matrix> logits = forwardPass(contextTokens, decoderTokens);

        // Calculate loss before backward pass
        int targetTokenId = m_Tokenizer->getTokenByName(targetToken);
        float loss = calculateLoss(logits, targetTokenId);
        // Update training statistics
        m_CurrentLoss = loss;
        m_TrainingSteps++;
        m_LossHistory.push_back(loss);
        if (m_LossHistory.size() > 1000) {
            m_LossHistory.erase(m_LossHistory.begin());
        }
        int predictedTokenId = predictToken(logits);
        std::string predictedToken = m_Tokenizer->getTokenById(predictedTokenId);
        // Debug output (only print occasionally to avoid spam)
        static int debugCounter = 0;
        if (++debugCounter % 100 == 0) {
            std::cout << "  [DEBUG] Loss: " << std::fixed << std::setprecision(4) << loss 
                      << " | Target: '" << targetToken << "' (ID:" << targetTokenId << ")"
                      << " | Predicted: '" << predictedToken << "' (ID:" << predictedTokenId << ")"
                      << " | Context: [";
            for (size_t i = 0; i < std::min(contextTokens.size(), (size_t)5); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << "'" << contextTokens[i] << "'";
            }
            if (contextTokens.size() > 5) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        // Backward pass
        std::shared_ptr<Matrix> targetMat = std::make_shared<Matrix>(1, m_VocabSize);
        for (int i = 0; i < m_VocabSize; ++i) (*targetMat)(0, i) = 0.0f;
        (*targetMat)(0, targetTokenId) = 1.0f;

        backwardPass(contextTokens, decoderTokens, targetMat, learningRate);
    }

    std::string Transformer::eval(const std::string& inputText) {
        // Tokenize input
        std::vector<std::string> inputTokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());
        
        // Prepare encoder input (the actual input text)
        std::vector<std::string> encInputTokens = inputTokens;
        while (encInputTokens.size() < m_SeqLen) encInputTokens.push_back("<PAD>");
        if (encInputTokens.size() > m_SeqLen) encInputTokens = std::vector<std::string>(encInputTokens.end() - m_SeqLen, encInputTokens.end());

        // Prepare decoder input (start with SOS, then PAD tokens)
        std::vector<std::string> decInputTokens(m_SeqLen, "<PAD>");
        decInputTokens[0] = "<SOS>";     // Start of generation
        
        // Convert to integer tokens for efficiency
        std::vector<int> encTokenIds = stringToTokenIds(encInputTokens);
        std::vector<int> decTokenIds = stringToTokenIds(decInputTokens);
        
        // Generate tokens iteratively
        std::vector<std::string> generatedTokens;
        int maxLength = m_SeqLen - 1; // Leave room for EOS
        int eosTokenId = getEosTokenId();
        
        for (int step = 0; step < maxLength; ++step) {
            // Forward pass
            auto logits = forwardPass(encInputTokens, decInputTokens);
            
            // Predict next token
            int nextTokenId = predictToken(logits);
            
            // Check for EOS
            if (nextTokenId == eosTokenId) {
                break;
            }
            
            // Get the token string
            std::string nextToken = m_Tokenizer->getTokenById(nextTokenId);
            generatedTokens.push_back(nextToken);
            
            // Update decoder input for next iteration
            // Shift tokens left and add the new token
            for (int i = 0; i < m_SeqLen - 1; ++i) {
                decTokenIds[i] = decTokenIds[i + 1];
            }
            decTokenIds[m_SeqLen - 1] = nextTokenId;
        }
        
        // Join generated tokens into a string
        std::string result;
        for (const auto& token : generatedTokens) {
            if (token != "<PAD>" && token != "<SOS>" && token != "<EOS>") {
                if (!result.empty()) {
                    result += " ";
                }
                result += token;
            }
        }
        
        return result;
    }

    std::string Transformer::evalWithTemperature(const std::string& inputText, float temperature, int maxLength) {
        // Tokenize input
        std::vector<std::string> inputTokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());
        
        // Prepare encoder input (the actual input text)
        std::vector<std::string> encInputTokens = inputTokens;
        while (encInputTokens.size() < m_SeqLen) encInputTokens.push_back("<PAD>");
        if (encInputTokens.size() > m_SeqLen) encInputTokens = std::vector<std::string>(encInputTokens.end() - m_SeqLen, encInputTokens.end());

        // Prepare decoder input (start with SOS, then PAD tokens)
        std::vector<std::string> decInputTokens(m_SeqLen, "<PAD>");
        decInputTokens[0] = "<SOS>";     // Start of generation
        
        // Convert to integer tokens for efficiency
        std::vector<int> encTokenIds = stringToTokenIds(encInputTokens);
        std::vector<int> decTokenIds = stringToTokenIds(decInputTokens);
        
        // Generate tokens iteratively
        std::vector<std::string> generatedTokens;
        int eosTokenId = getEosTokenId();
        
        for (int step = 0; step < maxLength; ++step) {
            // Forward pass
            auto logits = forwardPass(encInputTokens, decInputTokens);
            
            // Apply temperature and sample
            int nextTokenId = sampleTokenWithTemperature(logits, temperature);
            
            // Check for EOS
            if (nextTokenId == eosTokenId) {
                break;
            }
            
            // Get the token string
            std::string nextToken = m_Tokenizer->getTokenById(nextTokenId);
            generatedTokens.push_back(nextToken);
            
            // Update decoder input for next iteration
            // Shift tokens left and add the new token
            for (int i = 0; i < m_SeqLen - 1; ++i) {
                decTokenIds[i] = decTokenIds[i + 1];
            }
            decTokenIds[m_SeqLen - 1] = nextTokenId;
        }
        
        // Join generated tokens into a string
        std::string result;
        for (const auto& token : generatedTokens) {
            if (token != "<PAD>" && token != "<SOS>" && token != "<EOS>") {
                if (!result.empty()) {
                    result += " ";
                }
                result += token;
            }
        }
        
        return result;
    }

    int Transformer::sampleTokenWithTemperature(std::shared_ptr<Matrix> logits, float temperature) {
        int lastRow = logits->rows - 1;
        int padTokenId = getPadTokenId();

        // Apply temperature to logits, excluding PAD tokens
        std::vector<float> scaledLogits(m_VocabSize);
        for (int i = 0; i < m_VocabSize; ++i) {
            if (i == padTokenId) {
                scaledLogits[i] = -std::numeric_limits<float>::infinity(); // Set PAD to -inf
            } else {
                scaledLogits[i] = (*logits)(lastRow, i) / temperature;
            }
        }

        // Apply softmax to get probabilities
        std::vector<float> probabilities(m_VocabSize);
        float maxLogit = *std::max_element(scaledLogits.begin(), scaledLogits.end());

        float sum = 0.0f;
        for (int i = 0; i < m_VocabSize; ++i) {
            if (i == padTokenId) {
                probabilities[i] = 0.0f; // PAD tokens get zero probability
            } else {
                probabilities[i] = std::exp(scaledLogits[i] - maxLogit);
                sum += probabilities[i];
            }
        }

        // Normalize (only non-PAD tokens)
        if (sum > 0.0f) {
            for (int i = 0; i < m_VocabSize; ++i) {
                if (i != padTokenId) {
                    probabilities[i] /= sum;
                }
            }
        }

        // Sample from the distribution (excluding PAD)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        float randomValue = dis(gen);
        float cumulativeProb = 0.0f;

        for (int i = 0; i < m_VocabSize; ++i) {
            if (i == padTokenId) continue; // Skip PAD tokens

            cumulativeProb += probabilities[i];
            if (randomValue <= cumulativeProb) {
                return i;
            }
        }

        // Fallback to EOS if no valid token found
        return getEosTokenId();
    }

    void Transformer::resetPadTokenEmbedding() {
        // Reset PAD token embedding in the embedder
        m_Embedder->resetPadTokenEmbedding();
    }

    float Transformer::calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId) {
        // Calculate cross-entropy loss
        int lastRow = logits->rows - 1; // Get the last row (final prediction)

        // Apply softmax to get probabilities
        std::vector<float> probabilities(m_VocabSize);
        float maxLogit = (*logits)(lastRow, 0);

        // Find max for numerical stability
        for (int i = 1; i < m_VocabSize; ++i) {
            if ((*logits)(lastRow, i) > maxLogit) {
                maxLogit = (*logits)(lastRow, i);
            }
        }

        // Calculate softmax
        float sum = 0.0f;
        for (int i = 0; i < m_VocabSize; ++i) {
            probabilities[i] = std::exp((*logits)(lastRow, i) - maxLogit);
            sum += probabilities[i];
        }

        // Normalize
        for (int i = 0; i < m_VocabSize; ++i) {
            probabilities[i] /= sum;
        }

        // Calculate cross-entropy loss
        if (targetTokenId >= 0 && targetTokenId < m_VocabSize) {
            float targetProb = probabilities[targetTokenId];
            if (targetProb > 0.0f) {
                return -std::log(targetProb);
            } else {
                return 1000.0f; // High loss for zero probability
            }
        }

        return 1000.0f; // High loss for invalid target
    }

    void Transformer::resetTrainingStats() {
        m_LossHistory.clear();
        m_TrainingSteps = 0;
        m_CurrentLoss = 0.0f;
    }



    std::shared_ptr<Matrix> Transformer::forwardPass(std::vector<std::string>& encInputTokens, std::vector<std::string>& decInputTokens) {
        NNGL::Timer timer("Transformer::forwardPass");
        // 1. Get embeddings
        std::shared_ptr<Matrix> encInputMat = m_Embedder->forward(encInputTokens);
        m_Embedder->applyPositionalEncoding(encInputMat);
        std::shared_ptr<Matrix> decInputMat = m_Embedder->forward(decInputTokens);
        m_Embedder->applyPositionalEncoding(decInputMat);
        // 2. Create padding masks
        std::vector<int> encPaddingMask = createPaddingMask(stringToTokenIds(encInputTokens));
        std::vector<int> decPaddingMask = createPaddingMask(stringToTokenIds(decInputTokens));
        // 3. Forward through encoder
        std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat, encPaddingMask);
        // 4. Forward through decoder
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat, decPaddingMask, encPaddingMask);
        // 5. Extract only the last token's representation for next token prediction
        std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->cols);
        for (int i = 0; i < decOutputMat->cols; ++i) {
            (*lastTokenRep)(0, i) = (*decOutputMat)(decOutputMat->rows - 1, i);
        }
        // 6. Project decoder output to vocab logits
        return m_OutputProjection->forward(lastTokenRep);
    }

    void Transformer::backwardPass(const std::vector<std::string>& encInputTokens,
        const std::vector<std::string>& decInputTokens,
        std::shared_ptr<Matrix> targetMat,
        float learningRate) {
        NNGL::Timer timer("Transformer::backwardPass");
        // 1. Get embeddings
        std::shared_ptr<Matrix> encInputMat = m_Embedder->forward(const_cast<std::vector<std::string>&>(encInputTokens));
        m_Embedder->applyPositionalEncoding(encInputMat);
        std::shared_ptr<Matrix> decInputMat = m_Embedder->forward(const_cast<std::vector<std::string>&>(decInputTokens));
        m_Embedder->applyPositionalEncoding(decInputMat);
        // 2. Create padding masks
        std::vector<int> encPaddingMask = createPaddingMask(stringToTokenIds(encInputTokens));
        std::vector<int> decPaddingMask = createPaddingMask(stringToTokenIds(decInputTokens));
        // 3. Forward through encoder
        std::shared_ptr<Matrix> encOutputMat = m_Encoder->forward(encInputMat, encPaddingMask);
        // 4. Forward through decoder
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(decInputMat, encOutputMat, decPaddingMask, encPaddingMask);
        // 5. Backward through output projection
        std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->cols);
        for (int i = 0; i < decOutputMat->cols; ++i) {
            (*lastTokenRep)(0, i) = (*decOutputMat)(decOutputMat->rows - 1, i);
        }
        std::shared_ptr<Matrix> outputGrad = m_OutputProjection->backward(lastTokenRep, targetMat, learningRate);
        // 6. Expand gradient back to full sequence length for decoder backward pass
        std::shared_ptr<Matrix> decOutputGrad = std::make_shared<Matrix>(decOutputMat->rows, decOutputMat->cols);
        decOutputGrad->clear();
        for (int i = 0; i < decOutputMat->cols; ++i) {
            (*decOutputGrad)(decOutputMat->rows - 1, i) = (*outputGrad)(0, i);
        }
        // 7. Backward through decoder
        std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> decoderGrads =
            m_Decoder->backwardWithEncoderGrad(decOutputGrad, learningRate);
        std::shared_ptr<Matrix> decGrad = decoderGrads.first;
        std::shared_ptr<Matrix> encOutputGrad = decoderGrads.second;
        // Update decoder embeddings
        m_Embedder->removePositionalEncoding(decGrad);
        m_Embedder->backward(decInputTokens, decGrad, learningRate);
        // 8. Backward through encoder
        std::shared_ptr<Matrix> encGrad = m_Encoder->backward(encOutputGrad, learningRate);
        // Update encoder embeddings
        m_Embedder->removePositionalEncoding(encGrad);
        m_Embedder->backward(encInputTokens, encGrad, learningRate);
    }

    int Transformer::predictToken(std::shared_ptr<Matrix> probabilities) {
        int lastRow = probabilities->rows - 1;
        int padTokenId = getPadTokenId();

        int predictedToken = -1;
        float maxProb = -std::numeric_limits<float>::infinity();

        // Find the token with highest probability, excluding PAD tokens
        for (int i = 0; i < probabilities->cols; i++) {
            // Skip PAD tokens during prediction
            if (i == padTokenId) {
                continue;
            }
            
            float prob = (*probabilities)(lastRow, i);
            if (prob > maxProb) {
                maxProb = prob;
                predictedToken = i;
            }
        }
        
        // If no valid token found (shouldn't happen), return EOS
        if (predictedToken == -1) {
            predictedToken = getEosTokenId();
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
                tokenIds.push_back(getPadTokenId()); // Unknown token -> PAD
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

    // Helper methods for special token IDs
    int Transformer::getPadTokenId() const {
        return static_cast<int>(m_Tokenizer->getTokenByName("<PAD>"));
    }

    int Transformer::getSosTokenId() const {
        return static_cast<int>(m_Tokenizer->getTokenByName("<SOS>"));
    }

    int Transformer::getEosTokenId() const {
        return static_cast<int>(m_Tokenizer->getTokenByName("<EOS>"));
    }

    bool Transformer::isSpecialToken(int tokenId) const {
        return tokenId == getPadTokenId() || tokenId == getSosTokenId() || tokenId == getEosTokenId();
    }

    std::vector<int> Transformer::createPaddingMask(const std::vector<int>& tokenIds) const {
        std::vector<int> mask(tokenIds.size(), 1); // 1 for real tokens, 0 for PAD tokens
        int padTokenId = getPadTokenId();
        
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            if (tokenIds[i] == padTokenId) {
                mask[i] = 0; // Mark PAD tokens
            }
        }
        
        return mask;
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

    

    void Transformer::trainOnTokenSequence(const std::vector<std::string>& tokenSequence, float learningRate) {
        if (tokenSequence.size() < 2) {
            throw std::runtime_error("Need at least 2 tokens for next-token prediction");
        }
        trainNextToken(tokenSequence, learningRate);
    }
}
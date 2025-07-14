#include "GPTransformer.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cfloat>
#include <sstream>
#include <random>
#include <iomanip>
#include <limits>

namespace NNGL {
    GPTransformer::GPTransformer(std::string tokCheckpointFilepath, int modelDim, int hiddenDim, int seqLen) : m_SeqLen(seqLen) {
        m_Tokenizer = std::make_unique<BPE>();
        m_Tokenizer->load(tokCheckpointFilepath);
        m_Tokenizer->addToken("<PAD>");
        m_Tokenizer->addToken("<SOS>");
        m_Tokenizer->addToken("<EOS>");

        m_VocabSize = m_Tokenizer->getVocabSize();

        m_Embedder = std::make_unique<EmbeddingBlock>(m_VocabSize, modelDim, seqLen);
        m_Decoder = std::make_unique<DecoderBlock>(modelDim, hiddenDim, seqLen);
        m_OutputProjection = std::make_unique<NeuralNetwork>(1);
        m_OutputProjection->addLayer(modelDim, m_VocabSize, NNGL::ActivationFnType::IDENTITY);

        m_TrainingSteps = 0;
        m_CurrentLoss = 0.0f;
    }

    void GPTransformer::train(const std::string& inputText, float learningRate) {
        std::vector<std::string> tokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());
        if (tokens.size() < 2) return;
        tokens.push_back("<EOS>");
        trainOnSequence(tokens, 0, learningRate);
    }

    void GPTransformer::trainOnSequence(const std::vector<std::string>& longSequence, size_t windowSize, float learningRate) {
        if (windowSize == 0) windowSize = m_SeqLen + 1;
        if (longSequence.size() < windowSize) {
            trainNextToken(longSequence, learningRate);
            return;
        }
        for (size_t i = 0; i <= longSequence.size() - windowSize; ++i) {
            std::vector<std::string> window(
                longSequence.begin() + i,
                longSequence.begin() + i + windowSize
            );
            trainNextToken(window, learningRate);
        }
    }

    void GPTransformer::trainNextToken(const std::vector<std::string>& inputTokens, float learningRate) {
        if (inputTokens.size() < 2) throw std::runtime_error("Need at least 2 tokens for next-token prediction");
        std::vector<std::string> contextTokens(inputTokens.begin(), inputTokens.end() - 1);
        std::string targetToken = inputTokens.back();
        while (contextTokens.size() < m_SeqLen) contextTokens.push_back("<PAD>");
        if (contextTokens.size() > m_SeqLen) contextTokens = std::vector<std::string>(contextTokens.end() - m_SeqLen, contextTokens.end());
        std::shared_ptr<Matrix> logits = forwardPass(contextTokens);
        int targetTokenId = m_Tokenizer->getTokenByName(targetToken);
        float loss = calculateLoss(logits, targetTokenId, LossMode::Margin);
        m_CurrentLoss = loss;
        m_TrainingSteps++;
        m_LossHistory.push_back(loss);
        if (m_LossHistory.size() > 1000) m_LossHistory.erase(m_LossHistory.begin());
        int predictedTokenId = predictToken(logits);
        std::string predictedToken = m_Tokenizer->getTokenById(predictedTokenId);
        static int debugCounter = 0;
        if (++debugCounter % 14 == 0) {
            std::cout << "  Loss: " << std::fixed << std::setprecision(4) << loss 
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

        std::shared_ptr<Matrix> targetMat = std::make_shared<Matrix>(1, m_VocabSize, 0);
        targetMat->set(0, targetTokenId, 1.0f);

        backwardPass(contextTokens, targetMat, learningRate);
    }

    float GPTransformer::trainNextToken(const std::vector<std::string>& contextTokens, const std::string& targetToken, float learningRate) {
        // Fix padding logic: keep most recent tokens at the end, pad from the beginning
        std::vector<std::string> paddedContext = contextTokens;
        if (paddedContext.size() > m_SeqLen) {
            // Keep most recent tokens
            paddedContext = std::vector<std::string>(paddedContext.end() - m_SeqLen, paddedContext.end());
        }
        while (paddedContext.size() < m_SeqLen) {
            // Pad from the beginning (left side) - this is correct for causal LM
            paddedContext.insert(paddedContext.begin(), "<PAD>");
        }
        
        std::shared_ptr<Matrix> logits = forwardPass(paddedContext);
        int targetTokenId = m_Tokenizer->getTokenByName(targetToken);
        float loss = calculateLoss(logits, targetTokenId, LossMode::Margin);
        m_CurrentLoss = loss;
        m_TrainingSteps++;
        m_LossHistory.push_back(loss);
        if (m_LossHistory.size() > 1000) m_LossHistory.erase(m_LossHistory.begin());
        int predictedTokenId = predictToken(logits);
        std::string predictedToken = m_Tokenizer->getTokenById(predictedTokenId);
        static int debugCounter = 0;
        if (++debugCounter % 14 == 0) {
            std::cout << "  Loss: " << std::fixed << std::setprecision(4) << loss 
                      << " | Target: '" << targetToken << "' (ID:" << targetTokenId << ")"
                      << " | Predicted: '" << predictedToken << "' (ID:" << predictedTokenId << ")"
                      << " | Context: [";
            
            // Show the last 5 tokens (most recent context) instead of first 5
            size_t startIdx = (paddedContext.size() > 5) ? paddedContext.size() - 5 : 0;
            for (size_t i = startIdx; i < paddedContext.size(); ++i) {
                if (i > startIdx) std::cout << ", ";
                std::cout << "'" << paddedContext[i] << "'";
            }
            if (paddedContext.size() > 5) std::cout << " (last 5 of " << paddedContext.size() << ")";
            std::cout << "]" << std::endl;
        }

        std::shared_ptr<Matrix> targetMat = std::make_shared<Matrix>(1, m_VocabSize, 0);
        targetMat->set(0, targetTokenId, 1.0f);

        backwardPass(paddedContext, targetMat, learningRate);
        return loss;
    }

    std::string GPTransformer::eval(const std::string& inputText) {
        // 1. Tokenize input text
        std::vector<std::string> inputTokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());

        // 2. Build initial decoder input: <SOS> + input tokens + <PAD> ...
        std::vector<std::string> decInputTokens(m_SeqLen, "<PAD>");
        decInputTokens[0] = "<SOS>";
        int inputLen = std::min((int)inputTokens.size(), m_SeqLen - 1);
        for (int i = 0; i < inputLen; ++i) {
            decInputTokens[i + 1] = inputTokens[i];
        }

        std::vector<std::string> generatedTokens;
        int maxLength = m_SeqLen - 1;
        int eosTokenId = getEosTokenId();

        // 3. Generate tokens one by one, maintaining full context
        for (int step = 0; step < maxLength; ++step) {
            auto logits = forwardPass(decInputTokens);
            int nextTokenId = predictToken(logits);
            
            // Check for EOS
            if (nextTokenId == eosTokenId) break;
            
            std::string nextToken = m_Tokenizer->getTokenById(nextTokenId);
            generatedTokens.push_back(nextToken);
            
            // Maintain full context by shifting and adding new token
            // This preserves the causal nature of the model
            for (int i = 0; i < m_SeqLen - 1; ++i) {
                decInputTokens[i] = decInputTokens[i + 1];
            }
            decInputTokens[m_SeqLen - 1] = nextToken;
        }

        // 4. Build output string (skip special tokens)
        std::string result;
        for (const auto& token : generatedTokens) {
            if (token != "<PAD>" && token != "<SOS>" && token != "<EOS>") {
                result += token + "|";
            }
        }
        
        
        return result;
    }

    void GPTransformer::resetPadTokenEmbedding() {
        m_Embedder->resetPadTokenEmbedding();
    }

    float GPTransformer::calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId, LossMode mode) {
        if (std::isnan((*logits)(0, 0))) throw std::runtime_error("logits is nan(");

        std::vector<float> probabilities(m_VocabSize);
        float maxLogit = (*logits)(0, 0);
        for (int i = 1; i < m_VocabSize; ++i) {
            if ((*logits)(0, i) > maxLogit) maxLogit = (*logits)(0, i);
        }
        float sum = 0.0f;
        for (int i = 0; i < m_VocabSize; ++i) {
            probabilities[i] = std::exp((*logits)(0, i) - maxLogit);
            sum += probabilities[i];
        }
        for (int i = 0; i < m_VocabSize; ++i) probabilities[i] /= sum;

        switch (mode) {
            case LossMode::CrossEntropy: {
                if (targetTokenId >= 0 && targetTokenId < m_VocabSize) {
                    float targetProb = std::max(probabilities[targetTokenId], 1e-8f);
                    return -std::log(targetProb);
                }
                return 1000.0f;
            }
            case LossMode::Confidence: {
                if (targetTokenId >= 0 && targetTokenId < m_VocabSize) {
                    return probabilities[targetTokenId]; // [0, 1]
                }
                return 0.0f;
            }
            case LossMode::Margin: {
                float correctLogit = (*logits)(0, targetTokenId);
                float maxOtherLogit = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < m_VocabSize; ++i) {
                    if (i != targetTokenId && (*logits)(0, i) > maxOtherLogit) {
                        maxOtherLogit = (*logits)(0, i);
                    }
                }
                return 100 * (maxOtherLogit - correctLogit); // Higher is better
            }
            case LossMode::Accuracy: {
                // Argmax
                int predicted = 0;
                float maxVal = (*logits)(0, 0);
                for (int i = 1; i < m_VocabSize; ++i) {
                    if ((*logits)(0, i) > maxVal) {
                        maxVal = (*logits)(0, i);
                        predicted = i;
                    }
                }
                return (predicted == targetTokenId) ? 1.0f : 0.0f;
            }
            default:
                return 1000.0f;
        }
    }

    void GPTransformer::resetTrainingStats() {
        m_LossHistory.clear();
        m_TrainingSteps = 0;
        m_CurrentLoss = 0.0f;
    }

    std::shared_ptr<Matrix> GPTransformer::forwardPass(const std::vector<std::string>& inputTokens) {

        NNGL::Timer timer("GPTransformer::forwardPass");
        std::shared_ptr<Matrix> inputMat = m_Embedder->forward(inputTokens);

        int paddingLen = 0;
        std::vector<int> paddingMask = createPaddingMask(stringToTokenIds(inputTokens), paddingLen);
        m_Embedder->applyPositionalEncoding(inputMat, paddingMask);

        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(inputMat, paddingMask);
        std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->cols);
        for (int i = 0; i < decOutputMat->cols; ++i) (*lastTokenRep)(0, i) = (*decOutputMat)(decOutputMat->rows - 1, i);

        return m_OutputProjection->forward(lastTokenRep);
    }

    void GPTransformer::backwardPass(const std::vector<std::string>& inputTokens, std::shared_ptr<Matrix> targetMat, float learningRate) {

        NNGL::Timer timer("GPTransformer::backwardPass");

        std::shared_ptr<Matrix> inputMat = m_Embedder->forward(inputTokens);
        int paddingLen = 0;
        std::vector<int> paddingMask = createPaddingMask(stringToTokenIds(inputTokens), paddingLen);
        m_Embedder->applyPositionalEncoding(inputMat, paddingMask);

        // Use decoder-only architecture (no encoder, no cross-attention)
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(inputMat, paddingMask);
        std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->cols);
        for (int i = 0; i < decOutputMat->cols; ++i) (*lastTokenRep)(0, i) = (*decOutputMat)(decOutputMat->rows - 1, i);

        std::shared_ptr<Matrix> outputGrad = m_OutputProjection->backward(lastTokenRep, targetMat, learningRate);


        std::shared_ptr<Matrix> decOutputGrad = std::make_shared<Matrix>(decOutputMat->rows, decOutputMat->cols, 0.0f);
        int lastPos = decOutputMat->rows - 1;
        for (int i = 0; i < decOutputMat->cols; ++i) decOutputGrad->set(lastPos, i, outputGrad->get(0, i));
        std::shared_ptr<Matrix> decGrad = m_Decoder->backward(decOutputGrad, learningRate);

        m_Embedder->removePositionalEncoding(decGrad, paddingMask);
        m_Embedder->backward(inputTokens, decGrad, learningRate);
    }

    int GPTransformer::predictToken(std::shared_ptr<Matrix> probabilities) {
        int padTokenId = getPadTokenId();
        int predictedToken = -1;
        float maxProb = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < probabilities->cols; i++) {
            if (i == padTokenId) continue;
            float prob = (*probabilities)(0, i);
            if (prob > maxProb) {
                maxProb = prob;
                predictedToken = i;
            }
        }
        if (predictedToken == -1) predictedToken = getEosTokenId();
        return predictedToken;
    }

    std::vector<int> GPTransformer::stringToTokenIds(const std::vector<std::string>& tokens) {
        std::vector<int> tokenIds;
        tokenIds.reserve(tokens.size());
        for (const auto& token : tokens) {
            size_t tokenId = m_Tokenizer->getTokenByName(token);
            if (tokenId >= 0 && tokenId < m_VocabSize) tokenIds.push_back(static_cast<int>(tokenId));
            else tokenIds.push_back(getPadTokenId());
        }
        return tokenIds;
    }

    std::vector<std::string> GPTransformer::tokenIdsToStrings(const std::vector<int>& tokenIds) {
        std::vector<std::string> tokens;
        tokens.reserve(tokenIds.size());
        for (int tokenId : tokenIds) {
            if (tokenId >= 0 && tokenId < m_VocabSize) tokens.push_back(m_Tokenizer->getTokenById(tokenId));
            else tokens.push_back("<PAD>");
        }
        return tokens;
    }

    int GPTransformer::getPadTokenId() const {
        return static_cast<int>(m_Tokenizer->getTokenByName("<PAD>"));
    }
    int GPTransformer::getSosTokenId() const {
        return static_cast<int>(m_Tokenizer->getTokenByName("<SOS>"));
    }
    int GPTransformer::getEosTokenId() const {
        return static_cast<int>(m_Tokenizer->getTokenByName("<EOS>"));
    }
    bool GPTransformer::isSpecialToken(int tokenId) const {
        return tokenId == getPadTokenId() || tokenId == getSosTokenId() || tokenId == getEosTokenId();
    }
    std::vector<int> GPTransformer::createPaddingMask(const std::vector<int>& tokenIds, int &len) const {
        std::vector<int> mask(tokenIds.size(), 0);
        int padTokenId = getPadTokenId();
        
        // Find the first padding token to determine actual sequence length
        len = tokenIds.size(); // Default to full length
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            if (tokenIds[i] != padTokenId) {
                mask[i] = 1; // Mark as valid token
            } else {
                len = i; // Found first padding token
                break;
            }
        }
        
        return mask;
    }
    std::shared_ptr<Matrix> GPTransformer::getCachedEmbedding(const std::vector<int>& tokens) {
        std::stringstream ss;
        for (int tokenId : tokens) ss << tokenId << ",";
        std::string cacheKey = ss.str();
        {
            std::lock_guard<std::mutex> lock(m_CacheMutex);
            auto it = m_EmbeddingCache.find(cacheKey);
            if (it != m_EmbeddingCache.end()) return it->second;
        }
        std::vector<std::string> tokenStrings = tokenIdsToStrings(tokens);
        std::shared_ptr<Matrix> embedding = m_Embedder->forward(tokenStrings);
        {
            std::lock_guard<std::mutex> lock(m_CacheMutex);
            if (m_EmbeddingCache.size() > 1000) m_EmbeddingCache.clear();
            m_EmbeddingCache[cacheKey] = embedding;
        }
        return embedding;
    }
    void GPTransformer::printGradientHeatmap(std::shared_ptr<Matrix> mat) {
        const std::string colors[] = {
            "\033[48;5;17m", "\033[48;5;18m", "\033[48;5;19m", "\033[48;5;20m", "\033[48;5;21m",
            "\033[48;5;38m", "\033[48;5;44m", "\033[48;5;51m", "\033[48;5;87m", "\033[48;5;123m",
            "\033[48;5;159m", "\033[48;5;190m", "\033[48;5;226m", "\033[48;5;220m", "\033[48;5;202m",
            "\033[48;5;196m", "\033[0m"
        };
        constexpr int COLOR_COUNT = sizeof(colors) / sizeof(colors[0]) - 1;
        int rows = mat->rows;
        int cols = mat->cols;
        const int displaySize = 20;
        int rowStep = std::max(rows / displaySize, 1);
        int colStep = std::max(cols / displaySize, 1);
        std::vector<float> sampledValues;
        for (int i = 0; i < rows; i += rowStep) {
            for (int j = 0; j < cols; j += colStep) {
                sampledValues.push_back(mat->get(i, j));
            }
        }
        auto [minIt, maxIt] = std::minmax_element(sampledValues.begin(), sampledValues.end());
        float minVal = *minIt, maxVal = *maxIt;
        float range = (maxVal - minVal) > 1e-9f ? (maxVal - minVal) : 1.0f;
        size_t idx = 0;
        for (int i = 0; i < displaySize && i * rowStep < rows; ++i) {
            for (int j = 0; j < displaySize && j * colStep < cols; ++j) {
                float normalized = (sampledValues[idx++] - minVal) / range;
                int colorIdx = static_cast<int>(normalized * (COLOR_COUNT - 1));
                std::cout << colors[colorIdx] << "  " << colors[COLOR_COUNT];
            }
            std::cout << "\n";
        }
        std::cout << colors[COLOR_COUNT] << std::endl;
    }
    float GPTransformer::trainOnTokenSequence(const std::vector<std::string>& tokenSequence, float learningRate) {
        if (tokenSequence.size() < 2) throw std::runtime_error("Need at least 2 tokens for next-token prediction");
        trainNextToken(tokenSequence, learningRate);
        return m_CurrentLoss;
    }
} 
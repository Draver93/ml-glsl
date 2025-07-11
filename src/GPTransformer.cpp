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
        float loss = calculateLoss(logits, targetTokenId);
        m_CurrentLoss = loss;
        m_TrainingSteps++;
        m_LossHistory.push_back(loss);
        if (m_LossHistory.size() > 1000) m_LossHistory.erase(m_LossHistory.begin());
        int predictedTokenId = predictToken(logits);
        std::string predictedToken = m_Tokenizer->getTokenById(predictedTokenId);
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

        std::shared_ptr<Matrix> targetMat = std::make_shared<Matrix>(1, m_VocabSize, 0);
        targetMat->set(0, targetTokenId, 1.0f);

        backwardPass(contextTokens, targetMat, learningRate);
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
        std::vector<int> decTokenIds = stringToTokenIds(decInputTokens);


        std::vector<std::string> generatedTokens;
        int maxLength = m_SeqLen - 1;
        int eosTokenId = getEosTokenId();

        // 3. Start generation after the input context
        int contextLen = inputLen + 1; // <SOS> + input tokens
        for (int step = contextLen; step < maxLength; ++step) {
            auto logits = forwardPass(decInputTokens);
            int nextTokenId = predictToken(logits);
            if (nextTokenId == eosTokenId) break;
            std::string nextToken = m_Tokenizer->getTokenById(nextTokenId);
            generatedTokens.push_back(nextToken);
            
            // Shift left and append new token
            for (int i = 0; i < m_SeqLen - 1; ++i) decTokenIds[i] = decTokenIds[i + 1];
            decTokenIds[m_SeqLen - 1] = nextTokenId;
            decInputTokens = tokenIdsToStrings(decTokenIds);
        }

        // 4. Build output string (skip special tokens)
        std::string result;
        for (const auto& token : generatedTokens) {
            if (token != "<PAD>" && token != "<SOS>" && token != "<EOS>") {
                if (!result.empty()) result += " ";
                result += token;
            }
        }
        return result;
    }

    std::string GPTransformer::evalWithTemperature(const std::string& inputText, float temperature, int maxLength) {
        std::vector<std::string> inputTokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());
        std::vector<std::string> decInputTokens(m_SeqLen, "<PAD>");
        decInputTokens[0] = "<SOS>";
        std::vector<int> decTokenIds = stringToTokenIds(decInputTokens);
        std::vector<std::string> generatedTokens;
        int eosTokenId = getEosTokenId();
        for (int step = 0; step < maxLength; ++step) {
            auto logits = forwardPass(decInputTokens);
            int nextTokenId = sampleTokenWithTemperature(logits, temperature);
            if (nextTokenId == eosTokenId) break;
            std::string nextToken = m_Tokenizer->getTokenById(nextTokenId);
            generatedTokens.push_back(nextToken);
            for (int i = 0; i < m_SeqLen - 1; ++i) decTokenIds[i] = decTokenIds[i + 1];
            decTokenIds[m_SeqLen - 1] = nextTokenId;
            decInputTokens = tokenIdsToStrings(decTokenIds);
        }
        std::string result;
        for (const auto& token : generatedTokens) {
            if (token != "<PAD>" && token != "<SOS>" && token != "<EOS>") {
                if (!result.empty()) result += " ";
                result += token;
            }
        }
        return result;
    }

    int GPTransformer::sampleTokenWithTemperature(std::shared_ptr<Matrix> logits, float temperature) {
        int lastRow = logits->rows - 1;
        int padTokenId = getPadTokenId();
        std::vector<float> scaledLogits(m_VocabSize);
        for (int i = 0; i < m_VocabSize; ++i) {
            if (i == padTokenId) {
                scaledLogits[i] = -std::numeric_limits<float>::infinity();
            } else {
                scaledLogits[i] = (*logits)(lastRow, i) / temperature;
            }
        }
        std::vector<float> probabilities(m_VocabSize);
        float maxLogit = *std::max_element(scaledLogits.begin(), scaledLogits.end());
        float sum = 0.0f;
        for (int i = 0; i < m_VocabSize; ++i) {
            if (i == padTokenId) {
                probabilities[i] = 0.0f;
            } else {
                probabilities[i] = std::exp(scaledLogits[i] - maxLogit);
                sum += probabilities[i];
            }
        }
        if (sum > 0.0f) {
            for (int i = 0; i < m_VocabSize; ++i) {
                if (i != padTokenId) probabilities[i] /= sum;
            }
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float randomValue = dis(gen);
        float cumulativeProb = 0.0f;
        for (int i = 0; i < m_VocabSize; ++i) {
            if (i == padTokenId) continue;
            cumulativeProb += probabilities[i];
            if (randomValue <= cumulativeProb) return i;
        }
        return getEosTokenId();
    }

    void GPTransformer::resetPadTokenEmbedding() {
        m_Embedder->resetPadTokenEmbedding();
    }

    float GPTransformer::calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId) {
        int lastRow = logits->rows - 1;
        std::vector<float> probabilities(m_VocabSize);
        float maxLogit = (*logits)(lastRow, 0);
        for (int i = 1; i < m_VocabSize; ++i) {
            if ((*logits)(lastRow, i) > maxLogit) maxLogit = (*logits)(lastRow, i);
        }
        float sum = 0.0f;
        for (int i = 0; i < m_VocabSize; ++i) {
            probabilities[i] = std::exp((*logits)(lastRow, i) - maxLogit);
            sum += probabilities[i];
        }
        for (int i = 0; i < m_VocabSize; ++i) probabilities[i] /= sum;
        if (targetTokenId >= 0 && targetTokenId < m_VocabSize) {
            float targetProb = probabilities[targetTokenId];
            if (targetProb > 0.0f) return -std::log(targetProb);
            else return 1000.0f;
        }
        return 1000.0f;
    }

    void GPTransformer::resetTrainingStats() {
        m_LossHistory.clear();
        m_TrainingSteps = 0;
        m_CurrentLoss = 0.0f;
    }

    std::shared_ptr<Matrix> GPTransformer::forwardPass(std::vector<std::string>& inputTokens) {
        NNGL::Timer timer("GPTransformer::forwardPass");
        std::shared_ptr<Matrix> inputMat = m_Embedder->forward(inputTokens);
        int paddingLen = 0;
        std::vector<int> paddingMask = createPaddingMask(stringToTokenIds(inputTokens), paddingLen);
        m_Embedder->applyPositionalEncoding(inputMat);

        // Pass a dummy encoder output (zeros) to DecoderBlock
        std::shared_ptr<Matrix> dummyEncoderOutput = std::make_shared<Matrix>(m_SeqLen, inputMat->cols, 0.0f);
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(inputMat, dummyEncoderOutput, paddingMask, std::vector<int>(m_SeqLen, 1));
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
        std::shared_ptr<Matrix> dummyEncoderOutput = std::make_shared<Matrix>(m_SeqLen, inputMat->cols, 0.0f);
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(inputMat, dummyEncoderOutput, paddingMask, std::vector<int>(m_SeqLen, 1));
        std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->cols);
        for (int i = 0; i < decOutputMat->cols; ++i) (*lastTokenRep)(0, i) = (*decOutputMat)(decOutputMat->rows - 1, i);
        std::shared_ptr<Matrix> outputGrad = m_OutputProjection->backward(lastTokenRep, targetMat, learningRate);

        std::shared_ptr<Matrix> decOutputGrad = std::make_shared<Matrix>(decOutputMat->rows, decOutputMat->cols);
        decOutputGrad->clear();
        for (int i = 0; i < decOutputMat->cols; ++i) 
            decOutputGrad->set(paddingLen - 1, i, outputGrad->get(0, i)); // outputGrad 1 * modelDim

        std::shared_ptr<Matrix> decGrad = m_Decoder->backward(decOutputGrad, learningRate);
        m_Embedder->removePositionalEncoding(decGrad, paddingMask);
        m_Embedder->backward(inputTokens, decGrad, learningRate);
    }

    int GPTransformer::predictToken(std::shared_ptr<Matrix> probabilities) {
        int lastRow = probabilities->rows - 1;
        int padTokenId = getPadTokenId();
        int predictedToken = -1;
        float maxProb = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < probabilities->cols; i++) {
            if (i == padTokenId) continue;
            float prob = (*probabilities)(lastRow, i);
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
        bool lenFound = false;
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            if (tokenIds[i] != padTokenId) mask[i] = 1;
            else {
                len = i;
                return mask;
            }
        }
        len = tokenIds.size();
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
    void GPTransformer::trainOnTokenSequence(const std::vector<std::string>& tokenSequence, float learningRate) {
        if (tokenSequence.size() < 2) throw std::runtime_error("Need at least 2 tokens for next-token prediction");
        trainNextToken(tokenSequence, learningRate);
    }
} 
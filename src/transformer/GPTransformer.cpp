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
    GPTransformer::GPTransformer(std::string bpeFilepath, int modelDim, int hiddenDim, int seqLen) 
        : m_SeqLen(seqLen) {

        m_Embedder = std::make_unique<EmbeddingBlock>(bpeFilepath, modelDim, m_SeqLen);
        m_VocabSize = m_Embedder->getVocabSize();

        m_Decoder = std::make_unique<DecoderBlock>(modelDim, hiddenDim, seqLen);
        const char *data = m_Decoder->save();
        auto m_Decoder2 = std::make_unique<DecoderBlock>(data);



        m_OutputProjection = std::make_unique<NeuralNetwork>(1);
        m_OutputProjection->addLayer(modelDim, m_VocabSize, NNGL::ActivationFnType::IDENTITY);

        m_TargetMat = std::make_shared<Matrix>(1, m_VocabSize, 0);
        {
            std::vector<int> gradMask(seqLen, 0);
            gradMask[seqLen - 1] = 1;
            glGenBuffers(1, &m_GradMaskBuffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_GradMaskBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, gradMask.size() * sizeof(int), gradMask.data(), GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
    }

    GPTransformer::GPTransformer(std::string checkpointFilepath) {
        std::ifstream file(checkpointFilepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading: " + checkpointFilepath);


    }
    void GPTransformer::save(std::string checkpointFilepath) {
        std::ofstream file(checkpointFilepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + checkpointFilepath);

        int projection_nn_size = m_OutputProjection->getSaveSize();
        const char *projection_nn_buffer = m_OutputProjection->save();

        int embedder_size = m_Embedder->getSaveSize();
        const char* embedder_buffer = m_Embedder->save();

        int decoder_size = m_Decoder->getSaveSize();
        const char* decoder_buffer = m_Decoder->save();
    }


    float GPTransformer::trainNextToken(const std::vector<std::string>& contextTokens, const std::string& targetToken, float learningRate) {
        NNGL::Timer timer("GPTransformer::trainNextToken============================================================");

        std::vector<std::string> paddedContext = contextTokens;
        if (paddedContext.size() > m_SeqLen) paddedContext = std::vector<std::string>(paddedContext.end() - m_SeqLen, paddedContext.end());
        while (paddedContext.size() < m_SeqLen)  paddedContext.insert(paddedContext.begin(), "<PAD>");
        
        float loss = -1;
        int runCounter = 0;
        static int logCounter = 0;

        int targetTokenId = m_Embedder->getTokenByName(targetToken);
        m_TargetMat->clear();
        m_TargetMat->set(0, targetTokenId, 1.0f);
        m_TargetMat->uploadToGPU();

        logCounter++;
        if (0 && logCounter % 1000 == 0) {
            std::shared_ptr<Matrix> logits = forwardPass(paddedContext);
            logits->downloadFromGPU();
            loss = calculateLoss(logits, targetTokenId, LossMode::Margin);

            int predictedTokenId = predictToken(logits);
            std::string predictedToken = m_Embedder->getTokenById(predictedTokenId);

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
        {
            NNGL::Timer timer("GPTransformer::glFenceSync WAITING");
            GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

            backwardPass(paddedContext, m_TargetMat, learningRate);

            while (true) {
                GLenum result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000); // wait up to 0.1ms
                if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED)
                    break;
            }
            glDeleteSync(fence);
        }
 
        //if (++refreshCounter % 500 == 0) {
        //    std::this_thread::sleep_for(std::chrono::microseconds(5000000));  // ~0.5ms
        //}
        runCounter++;

        return loss;
    }

    std::string GPTransformer::eval(const std::string& inputText) {
        // 1. Tokenize input text
        std::vector<std::string> paddedContext = m_Embedder->tokenizeInput(inputText.data(), inputText.size());
        if (paddedContext.size() > m_SeqLen) {
            // Keep most recent tokens
            paddedContext = std::vector<std::string>(paddedContext.end() - m_SeqLen, paddedContext.end());
        }
        while (paddedContext.size() < m_SeqLen) {
            // Pad from the beginning (left side) - this is correct for causal LM
            paddedContext.insert(paddedContext.begin(), "<PAD>");
        }

        std::vector<std::string> generatedTokens;
        int maxLength = m_SeqLen - 1;
        int eosTokenId = getEosTokenId();

        // 3. Generate tokens one by one, maintaining full context
        for (int step = 0; step < maxLength; ++step) {
            auto logits = forwardPass(paddedContext);
            logits->downloadFromGPU();
            int nextTokenId = predictToken(logits);
            
            // Check for EOS
            if (nextTokenId == eosTokenId) break;
            
            std::string nextToken = m_Embedder->getTokenById(nextTokenId);
            generatedTokens.push_back(nextToken);
            
            // Maintain full context by shifting and adding new token
            // This preserves the causal nature of the model
            for (int i = 0; i < m_SeqLen - 1; ++i) {
                paddedContext[i] = paddedContext[i + 1];
            }
            paddedContext[m_SeqLen - 1] = nextToken;
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

    float GPTransformer::calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId, LossMode mode) {
        if (std::isnan((*logits)(0, 0))) throw std::runtime_error("logits is nan(");

        std::vector<float> probabilities(m_VocabSize);
        float maxLogit = (*logits)(0, 0);
        for (int i = 1; i < m_VocabSize; ++i) {
            if ((*logits)(i, 0) > maxLogit) maxLogit = (*logits)(i, 0);
        }
        float sum = 0.0f;
        for (int i = 0; i < m_VocabSize; ++i) {
            probabilities[i] = std::exp((*logits)(i, 0) - maxLogit);
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
                float correctLogit = (*logits)(targetTokenId, 0);
                float maxOtherLogit = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < m_VocabSize; ++i) {
                    if (i != targetTokenId && (*logits)(i, 0) > maxOtherLogit) {
                        maxOtherLogit = (*logits)(i, 0);
                    }
                }
                return 100 * (maxOtherLogit - correctLogit); // Higher is better
            }
            case LossMode::Accuracy: {
                // Argmax
                int predicted = 0;
                float maxVal = (*logits)(0, 0);
                for (int i = 1; i < m_VocabSize; ++i) {
                    if ((*logits)(i, 0) > maxVal) {
                        maxVal = (*logits)(i, 0);
                        predicted = i;
                    }
                }
                return (predicted == targetTokenId) ? 1.0f : 0.0f;
            }
            default:
                return 1000.0f;
        }
    }

    std::shared_ptr<Matrix> GPTransformer::forwardPass(const std::vector<std::string>& inputTokens) {

        NNGL::Timer timer("GPTransformer::forwardPass");
        std::shared_ptr<Matrix> inputMat = m_Embedder->forward(inputTokens);

        int paddingLen = 0;
        std::vector<int> paddingMask = createPaddingMask(stringToTokenIds(inputTokens), paddingLen);
        m_Embedder->applyPositionalEncoding(inputMat, paddingMask);

        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(inputMat, paddingMask);

        //std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->rows);
        //{
        //    NNGL::Timer timer("GPTransformer::forwardPass:1");
        //    decOutputMat->downloadFromGPU();
        //    for (int i = 0; i < decOutputMat->rows; ++i) (*lastTokenRep)(0, i) = (*decOutputMat)(i, decOutputMat->cols - 1);
        //    lastTokenRep->uploadToGPU();
        //} OLD cpu logic not we spec  decOutputMat->cols - 1

        return m_OutputProjection->forward(decOutputMat, decOutputMat->cols - 1);
    }

    void GPTransformer::backwardPass(const std::vector<std::string>& inputTokens, std::shared_ptr<Matrix> targetMat, float learningRate) {

        NNGL::Timer timer("GPTransformer::backwardPass");

        std::shared_ptr<Matrix> inputMat = m_Embedder->forward(inputTokens);
        int paddingLen = 0;
        std::vector<int> paddingMask = createPaddingMask(stringToTokenIds(inputTokens), paddingLen);
        m_Embedder->applyPositionalEncoding(inputMat, paddingMask);

        // Use decoder-only architecture (no encoder, no cross-attention)
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(inputMat, paddingMask);
        
        //std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->rows);
        //{
        //    NNGL::Timer timer("GPTransformer::backwardPass:1");
        //    decOutputMat->downloadFromGPU();
        //    for (int i = 0; i < decOutputMat->rows; ++i) (*lastTokenRep)(0, i) = (*decOutputMat)(i, decOutputMat->cols - 1);
        //    lastTokenRep->uploadToGPU();
        //} OLD cpu logic not we spec  decOutputMat->cols - 1
  
        std::shared_ptr<Matrix> outputGrad = m_OutputProjection->backward(decOutputMat, targetMat, learningRate, decOutputMat->cols - 1);

        std::shared_ptr<Matrix> decGrad = m_Decoder->backward(outputGrad, m_GradMaskBuffer, learningRate);
        m_Embedder->removePositionalEncoding(decGrad, paddingMask);
        m_Embedder->backward(inputTokens, decGrad, learningRate);
    }

    int GPTransformer::predictToken(std::shared_ptr<Matrix> probabilities) {
        int padTokenId = getPadTokenId();
        int predictedToken = -1;
        float maxProb = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < probabilities->rows; i++) {
            if (i == padTokenId) continue;
            float prob = (*probabilities)(i, 0);
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
            size_t tokenId = m_Embedder->getTokenByName(token);
            if (tokenId >= 0 && tokenId < m_VocabSize) tokenIds.push_back(static_cast<int>(tokenId));
            else tokenIds.push_back(getPadTokenId());
        }
        return tokenIds;
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
} 
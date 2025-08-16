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

namespace MLGL {
    GPTransformer::GPTransformer(std::string bpeFilepath, int modelDim, int hiddenDim, int seqLen) 
        : m_SeqLen(seqLen) {

        m_Embedder = std::make_unique<EmbeddingBlock>(bpeFilepath, modelDim, m_SeqLen);
        m_VocabSize = m_Embedder->getVocabSize();
        m_Decoder = std::make_unique<DecoderBlock>(modelDim, hiddenDim, seqLen);
        m_OutputProjection = std::make_unique<NeuralNetwork>(1);
        m_OutputProjection->addLayer(modelDim, m_VocabSize, MLGL::ActivationFnType::IDENTITY);

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

    // Complete constructor from checkpoint file
    GPTransformer::GPTransformer(std::string checkpointFilepath) {
        std::ifstream file(checkpointFilepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading: " + checkpointFilepath);

        // Read sequence length first
        file.read(reinterpret_cast<char*>(&m_SeqLen), sizeof(int));

        // Read vocab size
        file.read(reinterpret_cast<char*>(&m_VocabSize), sizeof(int));

        // Read sizes for each component
        int embedder_size, decoder_size, projection_nn_size;
        file.read(reinterpret_cast<char*>(&embedder_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&decoder_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&projection_nn_size), sizeof(int));

        // Load embedder
        char* embedder_buffer = new char[embedder_size];
        file.read(embedder_buffer, embedder_size);
        m_Embedder = std::make_unique<EmbeddingBlock>(embedder_buffer);
        delete[] embedder_buffer;

        // Load decoder
        char* decoder_buffer = new char[decoder_size];
        file.read(decoder_buffer, decoder_size);
        m_Decoder = std::make_unique<DecoderBlock>(decoder_buffer);
        delete[] decoder_buffer;

        // Load output projection
        char* projection_buffer = new char[projection_nn_size];
        file.read(projection_buffer, projection_nn_size);
        m_OutputProjection = std::make_unique<NeuralNetwork>(projection_buffer);
        delete[] projection_buffer;

        // Initialize target matrix
        m_TargetMat = std::make_shared<Matrix>(1, m_VocabSize, 0);

        // Initialize gradient mask buffer
        std::vector<int> gradMask(m_SeqLen, 0);
        gradMask[m_SeqLen - 1] = 1;
        glGenBuffers(1, &m_GradMaskBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_GradMaskBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gradMask.size() * sizeof(int), gradMask.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        file.close();
    }

    // Complete save method
    void GPTransformer::save(std::string checkpointFilepath) {
        std::ofstream file(checkpointFilepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + checkpointFilepath);

        // Write sequence length and vocab size first
        file.write(reinterpret_cast<const char*>(&m_SeqLen), sizeof(int));
        file.write(reinterpret_cast<const char*>(&m_VocabSize), sizeof(int));

        // Get serialized data from components
        int embedder_size = m_Embedder->getSaveSize();
        const char* embedder_buffer = m_Embedder->save();

        int decoder_size = m_Decoder->getSaveSize();
        const char* decoder_buffer = m_Decoder->save();

        int projection_nn_size = m_OutputProjection->getSaveSize();
        const char* projection_nn_buffer = m_OutputProjection->save();

        // Write sizes for each component
        file.write(reinterpret_cast<const char*>(&embedder_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&decoder_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&projection_nn_size), sizeof(int));

        // Write component data
        file.write(embedder_buffer, embedder_size);
        file.write(decoder_buffer, decoder_size);
        file.write(projection_nn_buffer, projection_nn_size);

        // Clean up temporary buffers
        delete[] embedder_buffer;
        delete[] decoder_buffer;
        delete[] projection_nn_buffer;

        file.close();
    }


    void GPTransformer::trainNextToken(const std::vector<std::string>& contextTokens, const std::string& targetToken, float learningRate) {
        MLGL::Timer timer("GPTransformer::trainNextToken============================================================", LogLevel::LL_DEBUG);

        std::vector<std::string> paddedContext = contextTokens;
        if (paddedContext.size() > m_SeqLen) paddedContext = std::vector<std::string>(paddedContext.end() - m_SeqLen, paddedContext.end());
        while (paddedContext.size() < m_SeqLen)  paddedContext.insert(paddedContext.begin(), "<PAD>");
        
        static int lossCalcInterval = 0;

        int targetTokenId = m_Embedder->getTokenByName(targetToken);
        m_TargetMat->clear();
        m_TargetMat->set(0, targetTokenId, 1.0f);
        m_TargetMat->uploadToGPU();

        lossCalcInterval++;
        if (lossCalcInterval % 10000 == 0) {
            std::shared_ptr<Matrix> logits = forwardPass(paddedContext);
            logits->downloadFromGPU();
            float loss = calculateLoss(logits, targetTokenId, LossMode::Margin);
        
            m_LossHistory.push_back(loss);
            while (m_LossHistory.size() > c_LossHistorySize) m_LossHistory.erase(m_LossHistory.begin());
        }

        {
            MLGL::Timer timer("GPTransformer::glFenceSync WAITING");

            backwardPass(paddedContext, m_TargetMat, learningRate);

            GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            while (true) {
                GLenum result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 100000);
                if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED)
                    break;
            }
            glDeleteSync(fence);
        }
    }

    std::string GPTransformer::eval(const std::string& inputText, bool include_sos) {
        // 1. Tokenize input text
        std::vector<std::string> paddedContext = m_Embedder->tokenizeInput(inputText.data(), inputText.size());
        if (include_sos) paddedContext.insert(paddedContext.begin(), "<SOS>");

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

        MLGL::Timer timer("GPTransformer::forwardPass");
        std::shared_ptr<Matrix> inputMat = m_Embedder->forward(inputTokens);

        std::vector<int> paddingMask = createPaddingMask(stringToTokenIds(inputTokens));
        // Find the first occurrence of target
        auto it = std::find(paddingMask.begin(), paddingMask.end(), 1);
        if (it == paddingMask.end()) std::cout << "Value not found in vector.\n";
        size_t index = std::distance(paddingMask.begin(), it);
        m_Embedder->applyPositionalEncoding(inputMat, index);


        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(inputMat, paddingMask);

        //std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->rows);
        //{
        //    MLGL::Timer timer("GPTransformer::forwardPass:1");
        //    decOutputMat->downloadFromGPU();
        //    for (int i = 0; i < decOutputMat->rows; ++i) (*lastTokenRep)(0, i) = (*decOutputMat)(i, decOutputMat->cols - 1);
        //    lastTokenRep->uploadToGPU();
        //} OLD cpu logic not we spec  decOutputMat->cols - 1

        return m_OutputProjection->forward(decOutputMat, decOutputMat->cols - 1);
    }

    void GPTransformer::backwardPass(const std::vector<std::string>& inputTokens, std::shared_ptr<Matrix> targetMat, float learningRate) {

        MLGL::Timer timer("GPTransformer::backwardPass");

        std::shared_ptr<Matrix> inputMat = m_Embedder->forward(inputTokens);

        std::vector<int> paddingMask = createPaddingMask(stringToTokenIds(inputTokens));
        // Find the first occurrence of target
        auto it = std::find(paddingMask.begin(), paddingMask.end(), 1);
        if (it == paddingMask.end()) std::cout << "Value not found in vector.\n";
        size_t index = std::distance(paddingMask.begin(), it);

        m_Embedder->applyPositionalEncoding(inputMat, index);

        // Use decoder-only architecture (no encoder, no cross-attention)
        std::shared_ptr<Matrix> decOutputMat = m_Decoder->forward(inputMat, paddingMask);
        
        //std::shared_ptr<Matrix> lastTokenRep = std::make_shared<Matrix>(1, decOutputMat->rows);
        //{
        //    MLGL::Timer timer("GPTransformer::backwardPass:1");
        //    decOutputMat->downloadFromGPU();
        //    for (int i = 0; i < decOutputMat->rows; ++i) (*lastTokenRep)(0, i) = (*decOutputMat)(i, decOutputMat->cols - 1);
        //    lastTokenRep->uploadToGPU();
        //} OLD cpu logic not we spec  decOutputMat->cols - 1
  
        std::shared_ptr<Matrix> outputGrad = m_OutputProjection->backward(decOutputMat, targetMat, learningRate, decOutputMat->cols - 1);

        std::shared_ptr<Matrix> decGrad = m_Decoder->backward(outputGrad, m_GradMaskBuffer, learningRate);
        m_Embedder->removePositionalEncoding(decGrad, index);
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

    std::vector<int> GPTransformer::createPaddingMask(const std::vector<int>& tokenIds) const {
        std::vector<int> mask(tokenIds.size(), 1);
        int padTokenId = getPadTokenId();
        
        // Find the first padding token to determine actual sequence length
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            if (tokenIds[i] == padTokenId) {
                mask[i] = 0; // Mark 
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
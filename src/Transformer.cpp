#include "Transformer.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cfloat>

namespace NNGL {
    Transformer::Transformer(std::string tokCheckpointFilepath, int modelDim, int hiddenDim, int seqLen) : m_SeqLen(seqLen) {
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

    void Transformer::train(const std::string& inputText) {
        // Tokenize the entire input
        std::vector<std::string> tokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());

        if (tokens.size() < 2) {
            //std::cerr << "Warning: Input too short for training" << std::endl;
            return;
        }

        // Train on sliding windows of the sequence
        trainOnSequence(tokens);
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

    std::string Transformer::eval(std::string& inputText) {
        std::vector<std::string> encInputTokens = m_Tokenizer->tokenizeInput(inputText.data(), inputText.size());
        while (encInputTokens.size() < m_SeqLen) encInputTokens.push_back("<PAD>");
        if (encInputTokens.size() > m_SeqLen) encInputTokens = std::vector<std::string>(encInputTokens.end() - m_SeqLen, encInputTokens.end());

        std::vector<std::string> decInputTokens(m_SeqLen, "<PAD>");
        decInputTokens.at(0) = "<SOS>";     // Start of generation

        int nextTokenId = predictToken(forwardPass(encInputTokens, decInputTokens));
        return m_Tokenizer->getTokenById(nextTokenId);
    }

    std::shared_ptr<Matrix> Transformer::forwardPass(std::vector<std::string>& encInputTokens, std::vector<std::string>& decInputTokens) {
        // 1. Embed input
        std::shared_ptr<Matrix> encInputMat = m_Embedder->forward(encInputTokens);
        m_Embedder->applyPositionalEncoding(encInputMat);

        std::shared_ptr<Matrix> decInputMat = m_Embedder->forward(decInputTokens);
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

    int Transformer::predictToken(std::shared_ptr<Matrix> logits) {
        int predictedToken = -1;
        float maxToken = FLT_MIN;
        for (int i = 0; i < logits->cols; i++) {
            if (maxToken < (*logits)(0, i)) {
                maxToken = (*logits)(0, i);
                predictedToken = i;
            }
        }
        return predictedToken;
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
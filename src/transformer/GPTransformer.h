#pragma once
#include "Logger.h"
#include "BPE.h"
#include "EmbeddingBlock.h"
#include "DecoderBlock.h"
#include "NeuralNetwork.h"
#include "Matrix.h"

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace MLGL {

    enum class LossMode { CrossEntropy, Confidence, Margin, Accuracy };

    class GPTransformer {
    public:
        GPTransformer(std::string bpeFilepath, int modelDim, int hiddenDim, int seqLen);
        GPTransformer(std::string checkpointFilepath);
        void save(std::string checkpointFilepath);

        void trainNextToken(const std::vector<std::string>& contextTokens, const std::string& targetToken, float learningRate);
        std::string eval(const std::string& inputText);

        float calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId, LossMode mode = LossMode::CrossEntropy);
        int predictToken(std::shared_ptr<Matrix> probabilities);

        std::vector<int> stringToTokenIds(const std::vector<std::string>& tokens);
        int getPadTokenId() const { return static_cast<int>(m_Embedder->getTokenByName("<PAD>")); }
        int getSosTokenId() const { return static_cast<int>(m_Embedder->getTokenByName("<SOS>")); }
        int getEosTokenId() const { return static_cast<int>(m_Embedder->getTokenByName("<EOS>")); }
        bool isSpecialToken(int tokenId) const { return tokenId == getPadTokenId() || tokenId == getSosTokenId() || tokenId == getEosTokenId(); }
        std::vector<std::string> tokenizeInput(const char* input, size_t inputLen) { return m_Embedder->tokenizeInput(input, inputLen); }
        float getAvrLoss() { if (m_LossHistory.empty()) return 0; float result = 0; for (auto& l : m_LossHistory) result += l; return result / m_LossHistory.size(); }

        std::vector<int> createPaddingMask(const std::vector<int>& tokenIds) const;
        void printGradientHeatmap(std::shared_ptr<Matrix> mat);
    private:
        std::shared_ptr<Matrix> forwardPass(const std::vector<std::string>& inputTokens);
        void backwardPass(const std::vector<std::string>& inputTokens, std::shared_ptr<Matrix> targetMat, float learningRate);

        std::unique_ptr<EmbeddingBlock> m_Embedder;
        std::unique_ptr<DecoderBlock> m_Decoder; 
        std::unique_ptr<NeuralNetwork> m_OutputProjection;

        int m_SeqLen;
        int m_VocabSize;
        std::shared_ptr<Matrix> m_TargetMat;
        GLuint m_GradMaskBuffer;

        const size_t c_LossHistorySize = 100;
        std::vector<float> m_LossHistory;
    };
} 
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

namespace NNGL {

    enum class LossMode { CrossEntropy, Confidence, Margin, Accuracy };

    class GPTransformer {
    public:
        GPTransformer(std::string bpeFilepath, int modelDim, int hiddenDim, int seqLen);
        GPTransformer(std::string checkpointFilepath);
        void save(std::string checkpointFilepath);

        float trainNextToken(const std::vector<std::string>& contextTokens, const std::string& targetToken, float learningRate);
        std::string eval(const std::string& inputText);

        float calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId, LossMode mode = LossMode::CrossEntropy);
        int predictToken(std::shared_ptr<Matrix> probabilities);

        std::vector<int> stringToTokenIds(const std::vector<std::string>& tokens);
        int getPadTokenId() const { return static_cast<int>(m_Embedder->getTokenByName("<PAD>")); }
        int getSosTokenId() const { return static_cast<int>(m_Embedder->getTokenByName("<SOS>")); }
        int getEosTokenId() const { return static_cast<int>(m_Embedder->getTokenByName("<EOS>")); }
        bool isSpecialToken(int tokenId) const { return tokenId == getPadTokenId() || tokenId == getSosTokenId() || tokenId == getEosTokenId(); }

        std::vector<int> createPaddingMask(const std::vector<int>& tokenIds, int& len) const;
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
    };
} 
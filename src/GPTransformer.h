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
        GPTransformer(std::string tokCheckpointFilepath, int modelDim, int hiddenDim, int seqLen);
        float trainNextToken(const std::vector<std::string>& contextTokens, const std::string& targetToken, float learningRate);
        std::string eval(const std::string& inputText);

        float calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId, LossMode mode = LossMode::CrossEntropy);
        void resetTrainingStats();
        int predictToken(std::shared_ptr<Matrix> probabilities);
        std::vector<int> stringToTokenIds(const std::vector<std::string>& tokens);
        std::vector<std::string> tokenIdsToStrings(const std::vector<int>& tokenIds);
        int getPadTokenId() const;
        int getSosTokenId() const;
        int getEosTokenId() const;
        bool isSpecialToken(int tokenId) const;
        std::vector<int> createPaddingMask(const std::vector<int>& tokenIds, int& len) const;
        std::shared_ptr<Matrix> getCachedEmbedding(const std::vector<int>& tokens);
        void printGradientHeatmap(std::shared_ptr<Matrix> mat);
        const std::vector<float>& getLossHistory() const { return m_LossHistory; }
        int getTrainingSteps() const { return m_TrainingSteps; }
        float getCurrentLoss() const { return m_CurrentLoss; }
    private:
        std::shared_ptr<Matrix> forwardPass(const std::vector<std::string>& inputTokens);
        void backwardPass(const std::vector<std::string>& inputTokens, std::shared_ptr<Matrix> targetMat, float learningRate);
        std::unique_ptr<BPE> m_Tokenizer;
        std::unique_ptr<EmbeddingBlock> m_Embedder;
        std::unique_ptr<DecoderBlock> m_Decoder; 
        std::unique_ptr<NeuralNetwork> m_OutputProjection;
        int m_VocabSize;
        int m_SeqLen;
        int m_TrainingSteps;
        float m_CurrentLoss;
        std::vector<float> m_LossHistory;
        std::mutex m_CacheMutex;
        std::unordered_map<std::string, std::shared_ptr<Matrix>> m_EmbeddingCache;
        std::shared_ptr<Matrix> m_TargetMat;
    };
} 
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
    class BPE;
    class EmbeddingBlock;
    class DecoderBlock;
    class NeuralNetwork;
    class Matrix;

    class GPTransformer {
    public:
        GPTransformer(std::string tokCheckpointFilepath, int modelDim, int hiddenDim, int seqLen);
        void train(const std::string& inputText, float learningRate);
        void trainOnSequence(const std::vector<std::string>& longSequence, size_t windowSize, float learningRate);
        void trainNextToken(const std::vector<std::string>& inputTokens, float learningRate);
        // New method for separate context and target
        float trainNextToken(const std::vector<std::string>& contextTokens, const std::string& targetToken, float learningRate);
        std::string eval(const std::string& inputText);
        std::string evalWithTemperature(const std::string& inputText, float temperature, int maxLength);
        void resetPadTokenEmbedding();
        float calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId);
        void resetTrainingStats();
        int sampleTokenWithTemperature(std::shared_ptr<Matrix> logits, float temperature);
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
        float trainOnTokenSequence(const std::vector<std::string>& tokenSequence, float learningRate);
            const std::vector<float>& getLossHistory() const { return m_LossHistory; }
    int getTrainingSteps() const { return m_TrainingSteps; }
    float getCurrentLoss() const { return m_CurrentLoss; }
    private:
        std::shared_ptr<Matrix> forwardPass(const std::vector<std::string>& inputTokens);
        void backwardPass(const std::vector<std::string>& inputTokens, std::shared_ptr<Matrix> targetMat, float learningRate);
        std::unique_ptr<BPE> m_Tokenizer;
        std::unique_ptr<EmbeddingBlock> m_Embedder;
        std::unique_ptr<DecoderOnlyBlock> m_Decoder;  // Changed from DecoderBlock to DecoderOnlyBlock
        std::unique_ptr<NeuralNetwork> m_OutputProjection;
        int m_VocabSize;
        int m_SeqLen;
        int m_TrainingSteps;
        float m_CurrentLoss;
        std::vector<float> m_LossHistory;
        std::mutex m_CacheMutex;
        std::unordered_map<std::string, std::shared_ptr<Matrix>> m_EmbeddingCache;
    };
} 
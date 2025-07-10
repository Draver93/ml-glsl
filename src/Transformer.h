#pragma once

#include "BPE.h"
#include "EmbeddingBlock.h"
#include "EncoderBlock.h"
#include "DecoderBlock.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace NNGL {
    class Transformer {
    public:
        Transformer(std::string tokCheckpointFilepath, int modelDim, int hiddenDim, int seqLen);

        void train(const std::string& inputText, float learningRate = 0.001f);
        std::string eval(const std::string& inputText);
        std::string evalWithTemperature(const std::string& inputText, float temperature = 1.0f, int maxLength = 50);

        // Training statistics
        float getCurrentLoss() const { return m_CurrentLoss; }
        int getTrainingSteps() const { return m_TrainingSteps; }
        const std::vector<float>& getLossHistory() const { return m_LossHistory; }
        void resetTrainingStats();
        void resetPadTokenEmbedding(); // Reset PAD token embedding to prevent explosion

        // Public method for training on specific token sequences
        void trainOnTokenSequence(const std::vector<std::string>& tokenSequence, float learningRate = 0.001f);

    private:
        void trainOnSequence(const std::vector<std::string>& longSequence, size_t windowSize = 0, float learningRate = 0.001f);
        void trainNextToken(const std::vector<std::string>& inputTokens, float learningRate);

        // Optimized versions using integer tokens
        void trainOnSequenceInt(const std::vector<int>& longSequence, size_t windowSize = 0, float learningRate = 0.001f);
        void trainNextTokenInt(const std::vector<int>& inputTokens, float learningRate);

        std::shared_ptr<Matrix> forwardPass(std::vector<std::string>& encInputTokens, std::vector<std::string>& decInputTokens);
        std::shared_ptr<Matrix> forwardPassInt(const std::vector<int>& encInputTokens, const std::vector<int>& decInputTokens);

        void backwardPass(const std::vector<std::string>& encInputTokens,
            const std::vector<std::string>& decInputTokens,
            std::shared_ptr<Matrix> targetMat,
            float learningRate);

        void backwardPassInt(const std::vector<int>& encInputTokens,
            const std::vector<int>& decInputTokens,
            int targetTokenId,
            float learningRate);

                int predictToken(std::shared_ptr<Matrix> probabilities);
        int sampleTokenWithTemperature(std::shared_ptr<Matrix> logits, float temperature);
        void printGradientHeatmap(std::shared_ptr<Matrix> mat);

        // Loss calculation
        float calculateLoss(std::shared_ptr<Matrix> logits, int targetTokenId);

        // Token conversion helpers
        std::vector<int> stringToTokenIds(const std::vector<std::string>& tokens);
        std::vector<std::string> tokenIdsToStrings(const std::vector<int>& tokenIds);

        // Special token helper methods
        int getPadTokenId() const;
        int getSosTokenId() const;
        int getEosTokenId() const;
        bool isSpecialToken(int tokenId) const;
        
        // Padding mask helper
        std::vector<int> createPaddingMask(const std::vector<int>& tokenIds) const;

        // Embedding caching
        std::unordered_map<std::string, std::shared_ptr<Matrix>> m_EmbeddingCache;
        std::mutex m_CacheMutex;
        std::shared_ptr<Matrix> getCachedEmbedding(const std::vector<int>& tokens);

    private:
        std::unique_ptr<BPE> m_Tokenizer;
        std::unique_ptr<EmbeddingBlock> m_Embedder;
        std::unique_ptr<EncoderBlock> m_Encoder;
        std::unique_ptr<DecoderBlock> m_Decoder;
        std::unique_ptr<NeuralNetwork> m_OutputProjection;

        int m_SeqLen;
        int m_VocabSize;

        // Training statistics
        std::vector<float> m_LossHistory;
        int m_TrainingSteps;
        float m_CurrentLoss;
        

    };
}
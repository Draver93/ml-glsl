#pragma once

#include "Matrix.h"
#include "Shader.h"
#include <random>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace NNGL {
    class EmbeddingBlock {
    private:
        std::mt19937 m_Generator;
        std::normal_distribution<float> m_Distribution;

        size_t m_VocabSize, m_ModelDim, m_MaxSeqLen;
        int m_ADAM_Timestep;
        std::unordered_map<std::string, std::vector<float>> m_Embeddings;
        
        // ADAM optimization buffers for embeddings
        std::unordered_map<std::string, std::vector<float>> m_ADAM_M_Embeddings;
        std::unordered_map<std::string, std::vector<float>> m_ADAM_V_Embeddings;

        std::shared_ptr<Matrix> m_PositionalEncodingMat;

        std::shared_ptr<Shader> m_ApplyPosEncodingCompute;
        std::shared_ptr<Shader> m_RemovePosEncodingCompute;
        std::shared_ptr<Shader> m_EmbeddingUpdateCompute; // NEW: embedding update shader

        // Training statistics
        size_t m_TotalUpdates;
        float m_AverageGradientMagnitude;
        std::vector<float> m_GradientHistory;

        std::shared_ptr<Matrix> m_CachedOutput; // Cached output matrix for forward()

        void initializePositionalEncoding();
        void initializeADAMBuffers(const std::string& token);
        std::vector<int> getTokenIndices(const std::vector<std::string>& tokens) const;

    public:
        EmbeddingBlock(size_t vocabSize, size_t modelDim, size_t maxSeqLen = 512);
        
        std::vector<float> initializeRandomVec();
        std::shared_ptr<Matrix> forward(const std::vector<std::string>& tokens);
        std::shared_ptr<Matrix> backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate);
        
        void applyPositionalEncoding(std::shared_ptr<Matrix> embeddings);
        void removePositionalEncoding(std::shared_ptr<Matrix> embeddings);
        void applyPositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask);
        void removePositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask);

        void save(const std::string& filename) const;
        void load(const std::string& filename);

        // Debugging and statistics
        void printEmbeddingStats() const;
        void printTokenEmbedding(const std::string& token) const;
        float getEmbeddingMagnitude(const std::string& token) const;
        size_t getVocabSize() const { return m_Embeddings.size(); }
        std::vector<std::string> getTopTokensByMagnitude(size_t count = 10) const;

        // Training statistics
        size_t getTotalUpdates() const { return m_TotalUpdates; }
        float getAverageGradientMagnitude() const { return m_AverageGradientMagnitude; }
        void resetTrainingStats();
        void resetPadTokenEmbedding(); // Reset PAD token embedding to prevent explosion
        std::shared_ptr<Matrix> getCachedOutput() { return m_CachedOutput; }
    };
} 
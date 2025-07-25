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
        std::shared_ptr<Matrix> m_EmbeddingsMat;
        std::unordered_map<std::string, int> m_EmbeddingsIds;

        
        std::mt19937 m_Generator;
        std::normal_distribution<float> m_Distribution;

        size_t m_VocabSize, m_ModelDim, m_MaxSeqLen;
        int m_ADAM_Timestep;
        
        // ADAM optimization buffers for embeddings (GPU)
        std::shared_ptr<Matrix> m_AdamMEmbeddings;
        std::shared_ptr<Matrix> m_AdamVEmbeddings;

        std::shared_ptr<Matrix> m_PositionalEncodingMat;

        std::shared_ptr<Shader> 
            m_ApplyPosEncodingCompute,
            m_RemovePosEncodingCompute,
            m_EmbeddingForwardCompute,
            m_EmbeddingUpdateCompute;

        // Training statistics
        size_t m_TotalUpdates;
        float m_AverageGradientMagnitude;
        std::vector<float> m_GradientHistory;

        GLuint m_CachedIndexBuffer = 0;

        void initializePositionalEncoding();

        std::shared_ptr<Matrix> m_CachedOutput; 
    private:
        GLuint getIndexBuffer(const std::vector<std::string>& tokens);
        GLuint getIndexBuffer(const std::vector<int>& indices);
    public:
        EmbeddingBlock(size_t vocabSize, size_t modelDim, size_t maxSeqLen);
        
        std::shared_ptr<Matrix> forward(const std::vector<std::string>& tokens);
        void backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate);
        
        void applyPositionalEncoding(std::shared_ptr<Matrix> embeddings);
        void removePositionalEncoding(std::shared_ptr<Matrix> embeddings);
        void applyPositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask);
        void removePositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask);
    };
} 
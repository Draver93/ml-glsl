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
        std::unordered_map<std::string, std::vector<float>> m_Embeddings;

        std::shared_ptr<Matrix> m_PositionalEncodingMat;

        std::shared_ptr<Shader> m_ApplyPosEncodingCompute;
        std::shared_ptr<Shader> m_RemovePosEncodingCompute;

        void initializePositionalEncoding();

    public:
        EmbeddingBlock(size_t vocabSize, size_t modelDim, size_t maxSeqLen = 512);
        
        std::vector<float> initializeRandomVec();
        std::shared_ptr<Matrix> forward(std::vector<std::string>& tokens);
        std::shared_ptr<Matrix> backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate);
        
        void applyPositionalEncoding(std::shared_ptr<Matrix> embeddings);
        void removePositionalEncoding(std::shared_ptr<Matrix> embeddings);
        
        void save(const std::string& filename) const;
        void load(const std::string& filename);
    };
} 
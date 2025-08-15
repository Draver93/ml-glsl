#pragma once

#include "Matrix.h"
#include "Shader.h"
#include <random>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "BPE.h"

namespace MLGL {
    class EmbeddingBlock {
    private:
        std::unique_ptr<BPE> m_Tokenizer;
        std::shared_ptr<Matrix> m_EmbeddingsMat;


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
        EmbeddingBlock(std::string bpeFilepath, size_t modelDim, size_t maxSeqLen);
        EmbeddingBlock(const char* data);
        int getSaveSize();
        const char* save();


        size_t getTokenByName(const std::string& name) { return m_Tokenizer->getTokenByName(name); }
        const std::string& getTokenById(int id) { return m_Tokenizer->getTokenById(id); }
        std::vector<std::string> tokenizeInput(const char* input, size_t inputLen) {  return m_Tokenizer->tokenizeInput(input, inputLen); }

        int getVocabSize() { return m_VocabSize; }

        std::shared_ptr<Matrix> forward(const std::vector<std::string>& tokens);
        void backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate);
        
        void applyPositionalEncoding(std::shared_ptr<Matrix> embeddings);
        void removePositionalEncoding(std::shared_ptr<Matrix> embeddings);
        void applyPositionalEncoding(std::shared_ptr<Matrix> embeddings, int first_token_idx);
        void removePositionalEncoding(std::shared_ptr<Matrix> embeddings, int first_token_idx);
    };
} 
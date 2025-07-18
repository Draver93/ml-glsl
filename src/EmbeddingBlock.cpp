#include "EmbeddingBlock.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unordered_map> // Added for unordered_map
#include "Logger.h"


namespace NNGL {
    EmbeddingBlock::EmbeddingBlock(size_t vocabSize, size_t modelDim, size_t maxSeqLen) :
        m_VocabSize(vocabSize), 
        m_ModelDim(modelDim),
        m_MaxSeqLen(maxSeqLen),
        m_ADAM_Timestep(0),
        m_TotalUpdates(0),
        m_AverageGradientMagnitude(0.0f) {


        // Initialize random number generator
        std::random_device rd;
        m_Generator.seed(rd());
        m_Distribution = std::normal_distribution<float>(0.0f, 0.1f);

        // Load compute shaders for positional encoding
        m_ApplyPosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/apply_pos_encoding.comp");
        m_RemovePosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/remove_pos_encoding.comp");

        m_EmbeddingUpdateCompute = ShaderManager::getInstance().getShader("shaders/embedding/embedding_update.comp");
        m_EmbeddingForwardCompute = ShaderManager::getInstance().getShader("shaders/embedding/embedding_forward.comp");

        initializePositionalEncoding();

        m_CachedOutput = std::make_shared<Matrix>(m_ModelDim, maxSeqLen);
        m_CachedOutput->uploadToGPU(); 

        m_EmbeddingsMat = std::make_shared<Matrix>(m_ModelDim, m_VocabSize);
        m_EmbeddingsMat->randomize(0.0f, 0.1f);
        m_EmbeddingsMat->uploadToGPU();
        m_EmbeddingsIds.reserve(m_VocabSize);

        // --- Adam buffers for embeddings ---
        m_AdamMEmbeddings = std::make_shared<Matrix>(m_ModelDim, m_VocabSize);
        m_AdamMEmbeddings->uploadToGPU();

        m_AdamVEmbeddings = std::make_shared<Matrix>(m_ModelDim, m_VocabSize);
        m_AdamVEmbeddings->uploadToGPU();
    }

    GLuint EmbeddingBlock::getIndexBuffer(const std::vector<std::string>& tokens) {
        std::vector<int> indices(m_MaxSeqLen, 0);
        for (int i = 0; i < tokens.size(); i++) {
            if (m_EmbeddingsIds.find(tokens[i]) == m_EmbeddingsIds.end()) {
                int id = m_EmbeddingsIds.size();
                m_EmbeddingsIds[tokens[i]] = id;
            }
            indices[i] = m_EmbeddingsIds[tokens[i]];
        }

        if (!m_CachedIndexBuffer) {
            glGenBuffers(1, &m_CachedIndexBuffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_CachedIndexBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_MaxSeqLen * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_CachedIndexBuffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, indices.size(), indices.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return m_CachedIndexBuffer;
    }


    std::shared_ptr<Matrix> EmbeddingBlock::forward(const std::vector<std::string>& tokens) {
        NNGL::Timer timer("EmbeddingBlock::forward");

        GLuint indexBuffer = getIndexBuffer(tokens);

        m_EmbeddingForwardCompute->bindBuffer(0, "EmbeddingsMat", DEBUG_VALIDATION(m_EmbeddingsMat));
        m_EmbeddingForwardCompute->bindBuffer(1, "Indices", indexBuffer);
        m_EmbeddingForwardCompute->bindBuffer(2, "OutputMat", m_CachedOutput->buffer);

        m_EmbeddingForwardCompute->setUniform("vocab_size", (int)m_VocabSize);
        m_EmbeddingForwardCompute->setUniform("model_dim", (int)m_ModelDim);
        m_EmbeddingForwardCompute->setUniform("max_seq_len", (int)m_MaxSeqLen);
        m_EmbeddingForwardCompute->setUniform("seq_len", (int)tokens.size());

        int localSizeX = 16;
        int localSizeY = 16;
        int workgroupsX = (m_ModelDim + localSizeX - 1) / localSizeX;
        int workgroupsY = (tokens.size() + localSizeY - 1) / localSizeY;
        m_EmbeddingForwardCompute->dispatch(workgroupsX, workgroupsY, 1);

        for (int j = 0; j <= 3; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
        DEBUG_VALIDATION(m_CachedOutput);
        return m_CachedOutput;
    }

    void EmbeddingBlock::backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate) {
        GLuint indexBuffer = getIndexBuffer(tokens);

        if (!gradOutput || gradOutput->rows != m_ModelDim) throw std::runtime_error("Invalid gradient dimensions");


        m_EmbeddingUpdateCompute->bindBuffer(0, "EmbeddingBuffer", DEBUG_VALIDATION(m_EmbeddingsMat));
        m_EmbeddingUpdateCompute->bindBuffer(1, "GradBuffer", DEBUG_VALIDATION(gradOutput));
        m_EmbeddingUpdateCompute->bindBuffer(2, "TokenIdxBuffer", indexBuffer);
        m_EmbeddingUpdateCompute->bindBuffer(3, "AdamMBuffer", m_AdamMEmbeddings->buffer);
        m_EmbeddingUpdateCompute->bindBuffer(4, "AdamVBuffer", m_AdamVEmbeddings->buffer);

        m_EmbeddingUpdateCompute->setUniform("vocab_size", (int)m_VocabSize);
        m_EmbeddingUpdateCompute->setUniform("model_dim", (int)m_ModelDim);
        m_EmbeddingUpdateCompute->setUniform("seq_len", (int)gradOutput->cols);
        m_EmbeddingUpdateCompute->setUniform("learning_rate", learningRate);
        m_EmbeddingUpdateCompute->setUniform("ADAM_beta1", 0.9f);
        m_EmbeddingUpdateCompute->setUniform("ADAM_beta2", 0.999f);
        m_EmbeddingUpdateCompute->setUniform("ADAM_timestep", m_ADAM_Timestep);

        int workgroupsX = (m_ModelDim + 15) / 16;
        m_EmbeddingUpdateCompute->dispatch(workgroupsX, 1, 1);
        DEBUG_VALIDATION(m_AdamMEmbeddings);
        DEBUG_VALIDATION(m_AdamVEmbeddings);

        for (int j = 0; j <= 4; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);

        m_ADAM_Timestep++;
    }

    void EmbeddingBlock::applyPositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask) {
        if (!embeddings || embeddings->rows != m_ModelDim) {
            throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding");
        }
        size_t seqLen = std::min(static_cast<size_t>(embeddings->cols), m_MaxSeqLen);

        // Upload padding mask buffer
        GLuint maskSSBO;
        glGenBuffers(1, &maskSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, maskSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, paddingMask.size() * sizeof(int), paddingMask.data(), GL_DYNAMIC_DRAW);
        // Bind buffers
        m_ApplyPosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", DEBUG_VALIDATION(embeddings));
        m_ApplyPosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", DEBUG_VALIDATION(m_PositionalEncodingMat));
        m_ApplyPosEncodingCompute->bindBuffer(2, "PaddingMask", maskSSBO);
        // Set uniforms
        m_ApplyPosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
        m_ApplyPosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));
        m_ApplyPosEncodingCompute->setUniform("has_padding_mask", true);
        // Dispatch compute shader
        int workgroupsX = (seqLen + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;
        m_ApplyPosEncodingCompute->dispatch(workgroupsX, workgroupsY, 1);

        glDeleteBuffers(1, &maskSSBO);

        for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }

    void EmbeddingBlock::applyPositionalEncoding(std::shared_ptr<Matrix> embeddings) {
        std::vector<int> mask(embeddings->rows, 1);
        applyPositionalEncoding(embeddings, mask);
    }

    void EmbeddingBlock::removePositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask) {
        if (!embeddings || embeddings->rows != m_ModelDim) {
            throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding removal");
        }
        size_t seqLen = std::min(static_cast<size_t>(embeddings->cols), m_MaxSeqLen);

        // Upload padding mask buffer
        GLuint maskSSBO;
        glGenBuffers(1, &maskSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, maskSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, paddingMask.size() * sizeof(int), paddingMask.data(), GL_DYNAMIC_DRAW);
        // Bind buffers
        m_RemovePosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", DEBUG_VALIDATION(embeddings));
        m_RemovePosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", DEBUG_VALIDATION(m_PositionalEncodingMat));
        m_RemovePosEncodingCompute->bindBuffer(2, "PaddingMask", maskSSBO);

        // Set uniforms
        m_RemovePosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
        m_RemovePosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));
        m_RemovePosEncodingCompute->setUniform("has_padding_mask", true);
        // Dispatch compute shader
        int workgroupsX = (seqLen + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;
        m_RemovePosEncodingCompute->dispatch(workgroupsX, workgroupsY, 1);

        glDeleteBuffers(1, &maskSSBO);
        // Unbind buffers
        for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }

    void EmbeddingBlock::removePositionalEncoding(std::shared_ptr<Matrix> embeddings) {
        std::vector<int> mask(embeddings->rows, 1);
        removePositionalEncoding(embeddings, mask);
    }

    void EmbeddingBlock::initializePositionalEncoding() {
        m_PositionalEncodingMat = std::make_shared<Matrix>(m_MaxSeqLen, m_ModelDim);

        for (size_t pos = 0; pos < m_MaxSeqLen; ++pos) {
            for (size_t i = 0; i < m_ModelDim; ++i) {
                float angle = pos / std::pow(10000.0f, 2.0f * (i / 2.0f) / m_ModelDim);
                if (i % 2 == 0) m_PositionalEncodingMat->set(pos, i, std::sin(angle));
                else m_PositionalEncodingMat->set(pos, i,std::cos(angle));
            }
        }

        m_PositionalEncodingMat->uploadToGPU();
    }
} 
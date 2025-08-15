#include "EmbeddingBlock.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unordered_map> 
#include "ActivationFunctions.h"

#include "Logger.h"


namespace MLGL {
    EmbeddingBlock::EmbeddingBlock(std::string bpeFilepath, size_t modelDim, size_t maxSeqLen) :
        m_ModelDim(modelDim),
        m_MaxSeqLen(maxSeqLen),
        m_ADAM_Timestep(0),
        m_TotalUpdates(0),
        m_AverageGradientMagnitude(0.0f) {

        m_Tokenizer = std::make_unique<BPE>();
        m_Tokenizer->load(bpeFilepath);
        m_Tokenizer->addToken("<PAD>");
        m_Tokenizer->addToken("<SOS>");
        m_Tokenizer->addToken("<EOS>");
        m_VocabSize = m_Tokenizer->getVocabSize();

        m_EmbeddingsMat = std::make_shared<Matrix>(m_VocabSize, m_ModelDim);
        m_EmbeddingsMat->randomize(0.0f, 0.1f);
        m_EmbeddingsMat->uploadToGPU();

        // Load compute shaders for positional encoding
        m_ApplyPosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/apply_pos_encoding.comp");
        m_RemovePosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/remove_pos_encoding.comp");

        m_EmbeddingUpdateCompute = ShaderManager::getInstance().getShader("shaders/embedding/embedding_update.comp");
        m_EmbeddingForwardCompute = ShaderManager::getInstance().getShader("shaders/embedding/embedding_forward.comp");

        initializePositionalEncoding();

        m_CachedOutput = std::make_shared<Matrix>(m_MaxSeqLen, m_ModelDim);
        m_CachedOutput->uploadToGPU(); 

        // --- Adam buffers for embeddings ---
        m_AdamMEmbeddings = std::make_shared<Matrix>(m_VocabSize, m_ModelDim);
        m_AdamMEmbeddings->uploadToGPU();

        m_AdamVEmbeddings = std::make_shared<Matrix>(m_VocabSize, m_ModelDim);
        m_AdamVEmbeddings->uploadToGPU();
    }

    EmbeddingBlock::EmbeddingBlock(const char* data) :
        m_ADAM_Timestep(0),
        m_TotalUpdates(0),
        m_AverageGradientMagnitude(0.0f) {

        if (!data) throw std::invalid_argument("Data pointer cannot be null");

        const char* ptr = data;

        // Read basic parameters
        std::memcpy(&m_ModelDim, ptr, sizeof(size_t));
        ptr += sizeof(size_t);

        std::memcpy(&m_MaxSeqLen, ptr, sizeof(size_t));
        ptr += sizeof(size_t);

        LOG_DEBUG("[EMBEDDING LOAD] Loading EmbeddingBlock with ModelDim=" + std::to_string(m_ModelDim) + ", MaxSeqLen=" + std::to_string(m_MaxSeqLen));

        {
            size_t tokenizer_size;
            std::memcpy(&tokenizer_size, ptr, sizeof(size_t));
            ptr += sizeof(size_t);
            // Load tokenizer from serialized data
            m_Tokenizer = std::make_unique<BPE>(ptr, tokenizer_size);
            ptr += tokenizer_size;

            m_VocabSize = m_Tokenizer->getVocabSize();
        }
        {
            // Load embeddings matrix
            size_t embeddings_data_size;
            std::memcpy(&embeddings_data_size, ptr, sizeof(size_t));
            ptr += sizeof(size_t);

            if (embeddings_data_size != m_VocabSize * m_ModelDim * sizeof(float)) {
                throw std::runtime_error("Embeddings data size mismatch during loading");
            }

            m_EmbeddingsMat = std::make_shared<Matrix>(m_VocabSize, m_ModelDim, reinterpret_cast<const float*>(ptr));
            ptr += embeddings_data_size;
            m_EmbeddingsMat->uploadToGPU();
        }

        LOG_DEBUG("[EMBEDDING LOAD] Tokenizer loaded, VocabSize=" + std::to_string(m_VocabSize));

        // Load compute shaders for positional encoding
        m_ApplyPosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/apply_pos_encoding.comp");
        m_RemovePosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/remove_pos_encoding.comp");

        m_EmbeddingUpdateCompute = ShaderManager::getInstance().getShader("shaders/embedding/embedding_update.comp");
        m_EmbeddingForwardCompute = ShaderManager::getInstance().getShader("shaders/embedding/embedding_forward.comp");

        initializePositionalEncoding();

        m_CachedOutput = std::make_shared<Matrix>(m_MaxSeqLen, m_ModelDim);
        m_CachedOutput->uploadToGPU();

        // --- Adam buffers for embeddings ---
        m_AdamMEmbeddings = std::make_shared<Matrix>(m_VocabSize, m_ModelDim);
        m_AdamMEmbeddings->uploadToGPU();

        m_AdamVEmbeddings = std::make_shared<Matrix>(m_VocabSize, m_ModelDim);
        m_AdamVEmbeddings->uploadToGPU();

        LOG_DEBUG("[EMBEDDING LOAD] EmbeddingBlock loaded successfully from binary buffer");
    }

    const char* EmbeddingBlock::save() {
        m_EmbeddingsMat->downloadFromGPU();

        LOG_DEBUG("[EMBEDDING SAVE] Saving EmbeddingBlock with ModelDim=" + std::to_string(m_ModelDim) +
            ", MaxSeqLen=" + std::to_string(m_MaxSeqLen));

        // Allocate buffer (caller is responsible for freeing this memory)
        char* buffer = new char[getSaveSize()];
        char* ptr = buffer;

        // Write basic parameters
        std::memcpy(ptr, &m_ModelDim, sizeof(size_t));
        ptr += sizeof(size_t);

        std::memcpy(ptr, &m_MaxSeqLen, sizeof(size_t));
        ptr += sizeof(size_t);

        // Save tokenizer
        size_t tokenizer_size = m_Tokenizer->getSaveSize();
        std::memcpy(ptr, &tokenizer_size, sizeof(size_t));
        ptr += sizeof(size_t);

        const char* tokenizer_data = m_Tokenizer->save();
        std::memcpy(ptr, tokenizer_data, tokenizer_size);
        ptr += tokenizer_size;

        // Write embeddings matrix
        size_t embeddings_size = m_EmbeddingsMat->byteSize();
        std::memcpy(ptr, &embeddings_size, sizeof(size_t));
        ptr += sizeof(size_t);

        std::memcpy(ptr, m_EmbeddingsMat->raw(), embeddings_size);
        ptr += embeddings_size;

        LOG_DEBUG("[EMBEDDING SAVE] Tokenizer data: " + std::to_string(tokenizer_size) + " bytes");
        LOG_DEBUG("[EMBEDDING SAVE] Embeddings: " + std::to_string(embeddings_size) + " bytes");

        return buffer;
    }

    int EmbeddingBlock::getSaveSize() {
        size_t header_size = 4 * sizeof(size_t); // basic parameters
        size_t tokenizer_size = m_Tokenizer->getSaveSize();
        size_t embeddings_size = m_EmbeddingsMat->byteSize();

        return static_cast<int>(header_size + tokenizer_size + embeddings_size);
    }

    GLuint EmbeddingBlock::getIndexBuffer(const std::vector<std::string>& tokens) {
        std::vector<int> indices(m_MaxSeqLen, 0);
        for (int i = 0; i < tokens.size(); i++) indices[i] = m_Tokenizer->getTokenByName(tokens[i]);

        return getIndexBuffer(indices);
    }

    GLuint EmbeddingBlock::getIndexBuffer(const std::vector<int>& indices) {
        if (!m_CachedIndexBuffer) {
            glGenBuffers(1, &m_CachedIndexBuffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_CachedIndexBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_MaxSeqLen * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_CachedIndexBuffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, indices.size() * sizeof(int), indices.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return m_CachedIndexBuffer;
    }

    std::shared_ptr<Matrix> EmbeddingBlock::forward(const std::vector<std::string>& tokens) {
        MLGL::Timer timer("EmbeddingBlock::forward");

        GLuint indexBuffer = getIndexBuffer(tokens);

        m_EmbeddingForwardCompute->bindBuffer(0, "EmbeddingsMat", m_EmbeddingsMat->buffer);
        m_EmbeddingForwardCompute->bindBuffer(1, "Indices", indexBuffer);
        m_EmbeddingForwardCompute->bindBuffer(2, "OutputMat", m_CachedOutput->buffer);

        m_EmbeddingForwardCompute->setUniform("vocab_size", (int)m_VocabSize);
        m_EmbeddingForwardCompute->setUniform("model_dim", (int)m_ModelDim);
        m_EmbeddingForwardCompute->setUniform("seq_len", (int)tokens.size());

        int localSizeX = 16;
        int localSizeY = 16;
        int workgroupsX = (m_ModelDim + localSizeX - 1) / localSizeX;
        int workgroupsY = (tokens.size() + localSizeY - 1) / localSizeY;
        m_EmbeddingForwardCompute->dispatch(workgroupsX, workgroupsY, 1);
        for (int j = 0; j <= 3; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);

        return m_CachedOutput;
    }

    void EmbeddingBlock::backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate) {
        if (!gradOutput || gradOutput->cols != m_ModelDim) 
            throw std::runtime_error("Invalid gradient dimensions");

        GLuint indexBuffer = getIndexBuffer(tokens);

        m_EmbeddingUpdateCompute->bindBuffer(0, "EmbeddingBuffer", m_EmbeddingsMat->buffer);
        m_EmbeddingUpdateCompute->bindBuffer(1, "GradBuffer", gradOutput->buffer);
        m_EmbeddingUpdateCompute->bindBuffer(2, "TokenIdxBuffer", indexBuffer);
        m_EmbeddingUpdateCompute->bindBuffer(3, "AdamMBuffer", m_AdamMEmbeddings->buffer);
        m_EmbeddingUpdateCompute->bindBuffer(4, "AdamVBuffer", m_AdamVEmbeddings->buffer);

        m_EmbeddingUpdateCompute->setUniform("vocab_size", (int)m_VocabSize);
        m_EmbeddingUpdateCompute->setUniform("model_dim", (int)m_ModelDim);
        m_EmbeddingUpdateCompute->setUniform("seq_len", (int)tokens.size());
        m_EmbeddingUpdateCompute->setUniform("learning_rate", learningRate);
        m_EmbeddingUpdateCompute->setUniform("ADAM_beta1", 0.9f);
        m_EmbeddingUpdateCompute->setUniform("ADAM_beta2", 0.999f);
        m_EmbeddingUpdateCompute->setUniform("ADAM_timestep", m_ADAM_Timestep);

        int workgroupsX = (m_ModelDim + 15) / 16;
        m_EmbeddingUpdateCompute->dispatch(workgroupsX, 1, 1);

        for (int j = 0; j <= 4; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);

        m_ADAM_Timestep++;
    }

    void EmbeddingBlock::applyPositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask) {
        if (!embeddings || embeddings->cols != m_ModelDim) 
            throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding");

        size_t seqLen = std::min(static_cast<size_t>(embeddings->rows), m_MaxSeqLen);
        GLuint indexBuffer = getIndexBuffer(paddingMask);

        // Bind buffers
        m_ApplyPosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", embeddings->buffer);
        m_ApplyPosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", m_PositionalEncodingMat->buffer);
        m_ApplyPosEncodingCompute->bindBuffer(2, "PaddingMask", indexBuffer);
        // Set uniforms
        m_ApplyPosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
        m_ApplyPosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));
        m_ApplyPosEncodingCompute->setUniform("has_padding_mask", true);
        // Dispatch compute shader
        int workgroupsX = (seqLen + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;

        m_ApplyPosEncodingCompute->dispatch(workgroupsX, workgroupsY, 1);

        for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }

    void EmbeddingBlock::applyPositionalEncoding(std::shared_ptr<Matrix> embeddings) {
        std::vector<int> mask(embeddings->rows, 1);
        applyPositionalEncoding(embeddings, mask);
    }

    void EmbeddingBlock::removePositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask) {
        if (!embeddings || embeddings->cols != m_ModelDim) 
            throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding removal");

        size_t seqLen = std::min(static_cast<size_t>(embeddings->rows), m_MaxSeqLen);

        GLuint indexBuffer = getIndexBuffer(paddingMask);

        // Bind buffers
        m_RemovePosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", embeddings->buffer);
        m_RemovePosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", m_PositionalEncodingMat->buffer);
        m_RemovePosEncodingCompute->bindBuffer(2, "PaddingMask", indexBuffer);

        // Set uniforms
        m_RemovePosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
        m_RemovePosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));
        m_RemovePosEncodingCompute->setUniform("has_padding_mask", true);
        // Dispatch compute shader
        int workgroupsX = (seqLen + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;
        m_RemovePosEncodingCompute->dispatch(workgroupsX, workgroupsY, 1);

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
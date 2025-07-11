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

        m_Embeddings.reserve(m_VocabSize);

        // Initialize random number generator
        std::random_device rd;
        m_Generator.seed(rd());
        m_Distribution = std::normal_distribution<float>(0.0f, 0.1f);

        // Initialize positional encoding matrix
        initializePositionalEncoding();

        // Load compute shaders for positional encoding
        m_ApplyPosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/apply_pos_encoding.comp");
        m_RemovePosEncodingCompute = ShaderManager::getInstance().getShader("shaders/embedding/remove_pos_encoding.comp");
        m_EmbeddingUpdateCompute = ShaderManager::getInstance().getShader("shaders/embedding/embedding_update.comp");
    }

    std::vector<float> EmbeddingBlock::initializeRandomVec() {
        std::vector<float> vec(m_ModelDim);

        for (int i = 0; i < m_ModelDim; i++) { 
            vec[i] = m_Distribution(m_Generator); 
        }
        return vec;
    }

    std::shared_ptr<Matrix> EmbeddingBlock::forward(const std::vector<std::string>& tokens) {
        NNGL::Timer timer("EmbeddingBlock::forward");
        size_t seqLen = tokens.size();
        // Reuse m_CachedOutput if possible
        if (!m_CachedOutput || m_CachedOutput->rows != seqLen || m_CachedOutput->cols != m_ModelDim) {
            m_CachedOutput = std::make_shared<Matrix>(seqLen, m_ModelDim);
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& t = tokens[i];
            if (m_Embeddings.find(t) == m_Embeddings.end()) {
                m_Embeddings[t] = initializeRandomVec();
            }
            for (size_t j = 0; j < m_ModelDim; ++j) {
                (*m_CachedOutput)(i, j) = m_Embeddings[t][j];
            }
        }
        return m_CachedOutput;
    }

    std::vector<int> EmbeddingBlock::getTokenIndices(const std::vector<std::string>& tokens) const {
        std::vector<int> indices;
        indices.reserve(tokens.size());
        for (const auto& token : tokens) {
            if (token == "<PAD>") {
                indices.push_back(-1); // skip PAD
            } else {
                auto it = m_Embeddings.find(token);
                if (it != m_Embeddings.end()) {
                    // Find index in embedding table
                    int idx = std::distance(m_Embeddings.begin(), m_Embeddings.find(token));
                    indices.push_back(idx);
                } else {
                    indices.push_back(-1);
                }
            }
        }
        return indices;
    }

    std::shared_ptr<Matrix> EmbeddingBlock::backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate) {
        NNGL::Timer timer("EmbeddingBlock::backward");
        if (!gradOutput || gradOutput->cols != m_ModelDim) {
            throw std::runtime_error("Invalid gradient dimensions");
        }
        // 1. Accumulate gradients for each unique token
        std::unordered_map<std::string, std::vector<float>> gradSums;
        for (size_t i = 0; i < tokens.size(); ++i) {
            const std::string& token = tokens[i];
            if (token == "<PAD>") continue;
            auto& sum = gradSums[token];
            if (sum.empty()) sum.resize(m_ModelDim, 0.0f);
            for (size_t j = 0; j < m_ModelDim; ++j) {
                sum[j] += (*gradOutput)(i, j);
            }
        }
        // 2. Prepare unique token list and gradient buffer
        std::vector<std::string> uniqueTokens;
        std::vector<float> gradBuffer;
        for (const auto& [token, grad] : gradSums) {
            uniqueTokens.push_back(token);
            gradBuffer.insert(gradBuffer.end(), grad.begin(), grad.end());
        }
        // 3. Prepare embedding buffer (flattened)
        std::vector<float> embeddingBuffer(m_VocabSize * m_ModelDim, 0.0f);
        std::vector<std::string> vocabTokens;
        vocabTokens.reserve(m_Embeddings.size());
        for (const auto& [token, vec] : m_Embeddings) {
            int idx = vocabTokens.size();
            vocabTokens.push_back(token);
            for (size_t j = 0; j < m_ModelDim; ++j) {
                embeddingBuffer[idx * m_ModelDim + j] = vec[j];
            }
        }
        // 4. Prepare token index buffer (indices into vocabTokens)
        std::vector<int> tokenIndices;
        for (const auto& token : uniqueTokens) {
            auto it = std::find(vocabTokens.begin(), vocabTokens.end(), token);
            if (it != vocabTokens.end()) {
                int idx = std::distance(vocabTokens.begin(), it);
                tokenIndices.push_back(idx);
            } else {
                tokenIndices.push_back(-1);
            }
        }

        // 5. Upload buffers to GPU
        GLuint embeddingSSBO;
        glGenBuffers(1, &embeddingSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, embeddingSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, embeddingBuffer.size() * sizeof(float), embeddingBuffer.data(), GL_DYNAMIC_DRAW);
        GLuint gradSSBO;
        glGenBuffers(1, &gradSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gradBuffer.size() * sizeof(float), gradBuffer.data(), GL_DYNAMIC_DRAW);
        GLuint tokenIdxSSBO;
        glGenBuffers(1, &tokenIdxSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, tokenIdxSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, tokenIndices.size() * sizeof(int), tokenIndices.data(), GL_DYNAMIC_DRAW);
        // Bind buffers
        m_EmbeddingUpdateCompute->bindBuffer(0, "EmbeddingBuffer", embeddingSSBO);
        m_EmbeddingUpdateCompute->bindBuffer(1, "GradBuffer", gradSSBO);
        m_EmbeddingUpdateCompute->bindBuffer(2, "TokenIdxBuffer", tokenIdxSSBO);
        m_EmbeddingUpdateCompute->setUniform("batch_size", (int)uniqueTokens.size());
        m_EmbeddingUpdateCompute->setUniform("vocab_size", (int)m_VocabSize);
        m_EmbeddingUpdateCompute->setUniform("model_dim", (int)m_ModelDim);
        m_EmbeddingUpdateCompute->setUniform("learning_rate", learningRate);
        // Dispatch
        int workgroupsX = (uniqueTokens.size() + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;
        m_EmbeddingUpdateCompute->dispatch(workgroupsX, workgroupsY, 1);
        // Download updated embeddings
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, embeddingSSBO);
        float* updatedEmbeds = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        for (size_t i = 0; i < vocabTokens.size(); ++i) {
            auto& vec = m_Embeddings[vocabTokens[i]];
            for (size_t j = 0; j < m_ModelDim; ++j) {
                vec[j] = updatedEmbeds[i * m_ModelDim + j];
            }
        }

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        // Cleanup
        glDeleteBuffers(1, &embeddingSSBO);
        glDeleteBuffers(1, &gradSSBO);
        glDeleteBuffers(1, &tokenIdxSSBO);
        // Return nullptr (no further backprop)
        return nullptr;
    }

    void EmbeddingBlock::applyPositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask) {
        if (!embeddings || embeddings->cols != m_ModelDim) {
            throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding");
        }
        size_t seqLen = std::min(static_cast<size_t>(embeddings->rows), m_MaxSeqLen);
        embeddings->uploadToGPU();
        m_PositionalEncodingMat->uploadToGPU();
        // Upload padding mask buffer
        GLuint maskSSBO;
        glGenBuffers(1, &maskSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, maskSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, paddingMask.size() * sizeof(int), paddingMask.data(), GL_DYNAMIC_DRAW);
        // Bind buffers
        m_ApplyPosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", embeddings->buffer);
        m_ApplyPosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", m_PositionalEncodingMat->buffer);
        m_ApplyPosEncodingCompute->bindBuffer(2, "PaddingMask", maskSSBO);
        // Set uniforms
        m_ApplyPosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
        m_ApplyPosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));
        // Dispatch compute shader
        int workgroupsX = (seqLen + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;
        m_ApplyPosEncodingCompute->dispatch(workgroupsX, workgroupsY, 1);
        embeddings->downloadFromGPU();
        glDeleteBuffers(1, &maskSSBO);
        // Unbind buffers
        for (int i = 0; i <= 2; ++i) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
    }
    // Overload for backward compatibility (no mask = all ones)
    void EmbeddingBlock::applyPositionalEncoding(std::shared_ptr<Matrix> embeddings) {
        std::vector<int> mask(embeddings->rows, 1);
        applyPositionalEncoding(embeddings, mask);
    }

    void EmbeddingBlock::removePositionalEncoding(std::shared_ptr<Matrix> embeddings, const std::vector<int>& paddingMask) {
        if (!embeddings || embeddings->cols != m_ModelDim) {
            throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding removal");
        }
        size_t seqLen = std::min(static_cast<size_t>(embeddings->rows), m_MaxSeqLen);
        embeddings->uploadToGPU();
        m_PositionalEncodingMat->uploadToGPU();
        // Upload padding mask buffer
        GLuint maskSSBO;
        glGenBuffers(1, &maskSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, maskSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, paddingMask.size() * sizeof(int), paddingMask.data(), GL_DYNAMIC_DRAW);
        // Bind buffers
        m_RemovePosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", embeddings->buffer);
        m_RemovePosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", m_PositionalEncodingMat->buffer);
        m_RemovePosEncodingCompute->bindBuffer(2, "PaddingMask", maskSSBO);
        // Set uniforms
        m_RemovePosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
        m_RemovePosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));
        // Dispatch compute shader
        int workgroupsX = (seqLen + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;
        m_RemovePosEncodingCompute->dispatch(workgroupsX, workgroupsY, 1);
        embeddings->downloadFromGPU();
        glDeleteBuffers(1, &maskSSBO);
        // Unbind buffers
        for (int i = 0; i <= 2; ++i) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
    }
    // Overload for backward compatibility (no mask = all ones)
    void EmbeddingBlock::removePositionalEncoding(std::shared_ptr<Matrix> embeddings) {
        std::vector<int> mask(embeddings->rows, 1);
        removePositionalEncoding(embeddings, mask);
    }

    void EmbeddingBlock::save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file for writing");

        // Write metadata
        size_t vocabSize = m_Embeddings.size();
        file.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));
        file.write(reinterpret_cast<const char*>(&m_ModelDim), sizeof(m_ModelDim));
        file.write(reinterpret_cast<const char*>(&m_MaxSeqLen), sizeof(m_MaxSeqLen));

        // Write embeddings
        for (const auto& [token, embedding] : m_Embeddings) {
            size_t tokenLength = token.length();
            file.write(reinterpret_cast<const char*>(&tokenLength), sizeof(tokenLength));
            file.write(token.c_str(), tokenLength);
            file.write(reinterpret_cast<const char*>(embedding.data()),
                embedding.size() * sizeof(float));
        }
    }

    void EmbeddingBlock::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file for reading");

        // Read metadata
        size_t vocabSize;
        size_t modelDim;
        size_t maxSeqLen;
        file.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
        file.read(reinterpret_cast<char*>(&modelDim), sizeof(modelDim));
        file.read(reinterpret_cast<char*>(&maxSeqLen), sizeof(maxSeqLen));

        if (modelDim != m_ModelDim) {
            throw std::runtime_error("Model dimension mismatch");
        }

        if (maxSeqLen != m_MaxSeqLen) {
            m_MaxSeqLen = maxSeqLen;
            // Reinitialize positional encoding with new sequence length
            initializePositionalEncoding();
        }

        m_Embeddings.clear();
        m_Embeddings.reserve(vocabSize);

        for (size_t i = 0; i < vocabSize; i++) {
            size_t tokenLength;
            file.read(reinterpret_cast<char*>(&tokenLength), sizeof(tokenLength));

            std::string token(tokenLength, '\0');
            file.read(&token[0], tokenLength);

            std::vector<float> embedding(m_ModelDim);
            file.read(reinterpret_cast<char*>(embedding.data()),
                m_ModelDim * sizeof(float));

            m_Embeddings[token] = std::move(embedding);
        }
    }

    void EmbeddingBlock::initializePositionalEncoding() {
        // Create positional encoding matrix [max_seq_len, model_dim]
        m_PositionalEncodingMat = std::make_shared<Matrix>(m_MaxSeqLen, m_ModelDim);

        // Initialize with sinusoidal positional encoding
        for (size_t pos = 0; pos < m_MaxSeqLen; ++pos) {
            for (size_t i = 0; i < m_ModelDim; ++i) {
                float angle = pos / std::pow(10000.0f, 2.0f * (i / 2.0f) / m_ModelDim);
                if (i % 2 == 0) {
                    (*m_PositionalEncodingMat)(pos, i) = std::sin(angle);
                }
                else {
                    (*m_PositionalEncodingMat)(pos, i) = std::cos(angle);
                }
            }
        }

        // Upload to GPU
        m_PositionalEncodingMat->uploadToGPU();
    }

    void EmbeddingBlock::initializeADAMBuffers(const std::string& token) {
        if (m_ADAM_M_Embeddings.find(token) == m_ADAM_M_Embeddings.end()) {
            m_ADAM_M_Embeddings[token] = std::vector<float>(m_ModelDim, 0.0f);
            m_ADAM_V_Embeddings[token] = std::vector<float>(m_ModelDim, 0.0f);
        }
    }

    void EmbeddingBlock::printEmbeddingStats() const {
        std::cout << "=== Embedding Statistics ===" << std::endl;
        std::cout << "Vocabulary size: " << m_Embeddings.size() << std::endl;
        std::cout << "Model dimension: " << m_ModelDim << std::endl;
        std::cout << "Total updates: " << m_TotalUpdates << std::endl;
        std::cout << "Average gradient magnitude: " << std::fixed << std::setprecision(4) << m_AverageGradientMagnitude << std::endl;

        if (!m_GradientHistory.empty()) {
            float recentAvg = 0.0f;
            int count = std::min(10, (int)m_GradientHistory.size());
            for (int i = m_GradientHistory.size() - count; i < m_GradientHistory.size(); ++i) {
                recentAvg += m_GradientHistory[i];
            }
            recentAvg /= count;
            std::cout << "Recent gradient magnitude (last " << count << "): " << std::fixed << std::setprecision(4) << recentAvg << std::endl;
        }
    }

    void EmbeddingBlock::printTokenEmbedding(const std::string& token) const {
        auto it = m_Embeddings.find(token);
        if (it == m_Embeddings.end()) {
            std::cout << "Token '" << token << "' not found in vocabulary" << std::endl;
            return;
        }

        const auto& embedding = it->second;
        float magnitude = 0.0f;
        for (float val : embedding) {
            magnitude += val * val;
        }
        magnitude = std::sqrt(magnitude);

        std::cout << "Token: '" << token << "' | Magnitude: " << std::fixed << std::setprecision(4) << magnitude << std::endl;
        std::cout << "First 10 values: [";
        for (size_t i = 0; i < std::min(embedding.size(), (size_t)10); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(3) << embedding[i];
        }
        std::cout << "]" << std::endl;
    }

    float EmbeddingBlock::getEmbeddingMagnitude(const std::string& token) const {
        auto it = m_Embeddings.find(token);
        if (it == m_Embeddings.end()) {
            return 0.0f;
        }

        const auto& embedding = it->second;
        float magnitude = 0.0f;
        for (float val : embedding) {
            magnitude += val * val;
        }
        return std::sqrt(magnitude);
    }

    std::vector<std::string> EmbeddingBlock::getTopTokensByMagnitude(size_t count) const {
        std::vector<std::pair<std::string, float>> tokenMagnitudes;

        for (const auto& [token, embedding] : m_Embeddings) {
            float magnitude = getEmbeddingMagnitude(token);
            tokenMagnitudes.emplace_back(token, magnitude);
        }

        // Sort by magnitude (descending)
        std::sort(tokenMagnitudes.begin(), tokenMagnitudes.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Return top tokens
        std::vector<std::string> result;
        for (size_t i = 0; i < std::min(count, tokenMagnitudes.size()); ++i) {
            result.push_back(tokenMagnitudes[i].first);
        }

        return result;
    }

    void EmbeddingBlock::resetTrainingStats() {
        m_TotalUpdates = 0;
        m_AverageGradientMagnitude = 0.0f;
        m_GradientHistory.clear();
    }

    void EmbeddingBlock::resetPadTokenEmbedding() {
        auto it = m_Embeddings.find("<PAD>");
        if (it != m_Embeddings.end()) {
            // Reset PAD token embedding to small random values
            for (size_t i = 0; i < m_ModelDim; ++i) {
                it->second[i] = m_Distribution(m_Generator) * 0.1f; // Small random values
            }

            // Reset ADAM buffers for PAD token
            m_ADAM_M_Embeddings.erase("<PAD>");
            m_ADAM_V_Embeddings.erase("<PAD>");

            std::cout << "  [EMBED] Reset PAD token embedding to small values" << std::endl;
        }
    }
} 
#include "EmbeddingBlock.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace NNGL {
    EmbeddingBlock::EmbeddingBlock(size_t vocabSize, size_t modelDim, size_t maxSeqLen) :
        m_VocabSize(vocabSize), 
        m_ModelDim(modelDim),
        m_MaxSeqLen(maxSeqLen),
        m_ADAM_Timestep(0) {

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
    }

    std::vector<float> EmbeddingBlock::initializeRandomVec() {
        std::vector<float> vec(m_ModelDim);

        for (int i = 0; i < m_ModelDim; i++) { 
            vec[i] = m_Distribution(m_Generator); 
        }
        return vec;
    }

    std::shared_ptr<Matrix> EmbeddingBlock::forward(std::vector<std::string>& tokens) {
        std::vector<std::vector<float>> tmpVec; 
        tmpVec.reserve(tokens.size());

        for (auto& t : tokens) {
            if (m_Embeddings.find(t) == m_Embeddings.end()) {
                m_Embeddings[t] = initializeRandomVec();
            }
            tmpVec.push_back(m_Embeddings[t]);
        }
        return std::make_shared<Matrix>(tmpVec);
    }

    std::shared_ptr<Matrix> EmbeddingBlock::backward(const std::vector<std::string>& tokens, std::shared_ptr<Matrix> gradOutput, float learningRate) {
        if (!gradOutput || gradOutput->cols != m_ModelDim) {
            throw std::runtime_error("Invalid gradient dimensions");
        }

        // Download gradients from GPU
        gradOutput->downloadFromGPU();

        // ADAM parameters
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1e-8f;

        // Update embeddings using ADAM optimization
        size_t minSize = std::min(tokens.size(), static_cast<size_t>(gradOutput->rows));

        for (size_t i = 0; i < minSize; ++i) {
            const std::string& token = tokens[i];
            auto it = m_Embeddings.find(token);

            if (it != m_Embeddings.end()) {
                // Initialize ADAM buffers if needed
                initializeADAMBuffers(token);
                
                auto& embedding = it->second;
                auto& adamM = m_ADAM_M_Embeddings[token];
                auto& adamV = m_ADAM_V_Embeddings[token];

                for (size_t j = 0; j < m_ModelDim; ++j) {
                    float gradient = (*gradOutput)(i, j);
                    
                    // ADAM update
                    adamM[j] = beta1 * adamM[j] + (1.0f - beta1) * gradient;
                    adamV[j] = beta2 * adamV[j] + (1.0f - beta2) * gradient * gradient;
                    
                    // Bias correction
                    float m_hat = adamM[j] / (1.0f - std::pow(beta1, m_ADAM_Timestep + 1));
                    float v_hat = adamV[j] / (1.0f - std::pow(beta2, m_ADAM_Timestep + 1));
                    
                    // Update embedding
                    embedding[j] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
        }

        // Increment ADAM timestep
        m_ADAM_Timestep++;

        // Return the gradient (no further backprop beyond embeddings)
        return nullptr;
    }

    void EmbeddingBlock::applyPositionalEncoding(std::shared_ptr<Matrix> embeddings) {
        if (!embeddings || embeddings->cols != m_ModelDim) {
            throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding");
        }

        size_t seqLen = std::min(static_cast<size_t>(embeddings->rows), m_MaxSeqLen);

        // Upload matrices to GPU
        embeddings->uploadToGPU();
        m_PositionalEncodingMat->uploadToGPU();

        // Bind buffers
        m_ApplyPosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", embeddings->buffer);
        m_ApplyPosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", m_PositionalEncodingMat->buffer);

        // Set uniforms
        m_ApplyPosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
        m_ApplyPosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));

        // Dispatch compute shader
        int workgroupsX = (seqLen + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;
        m_ApplyPosEncodingCompute->dispatch(workgroupsX, workgroupsY, 1);

        embeddings->downloadFromGPU();

        // Unbind buffers
        for (int i = 0; i <= 1; ++i) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
    }

    void EmbeddingBlock::removePositionalEncoding(std::shared_ptr<Matrix> embeddings) {
        if (!embeddings || embeddings->cols != m_ModelDim) {
            throw std::runtime_error("Invalid embedding matrix dimensions for positional encoding removal");
        }

        size_t seqLen = std::min(static_cast<size_t>(embeddings->rows), m_MaxSeqLen);

        // Upload matrices to GPU
        embeddings->uploadToGPU();
        m_PositionalEncodingMat->uploadToGPU();

        // Bind buffers
        m_RemovePosEncodingCompute->bindBuffer(0, "EmbeddingsBuffer", embeddings->buffer);
        m_RemovePosEncodingCompute->bindBuffer(1, "PositionalEncodingBuffer", m_PositionalEncodingMat->buffer);

        // Set uniforms
        m_RemovePosEncodingCompute->setUniform("seq_len", static_cast<int>(seqLen));
        m_RemovePosEncodingCompute->setUniform("model_dim", static_cast<int>(m_ModelDim));

        // Dispatch compute shader
        int workgroupsX = (seqLen + 15) / 16;
        int workgroupsY = (m_ModelDim + 15) / 16;
        m_RemovePosEncodingCompute->dispatch(workgroupsX, workgroupsY, 1);

        embeddings->downloadFromGPU();

        // Unbind buffers
        for (int i = 0; i <= 1; ++i) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
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
} 
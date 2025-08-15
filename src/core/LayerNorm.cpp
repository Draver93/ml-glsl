#include "LayerNorm.h"
#include "Logger.h"
#include <cmath>
#include <algorithm>

namespace MLGL {
    LayerNorm::LayerNorm(int normalizedShape, float epsilon)
        : m_NormalizedShape(normalizedShape), m_Epsilon(epsilon) {

        m_Gamma = std::make_shared<Matrix>(normalizedShape, 1, 1.0f);
        m_Gamma->uploadToGPU();

        m_Beta = std::make_shared<Matrix>(normalizedShape, 1, 0.0f);
        m_Beta->uploadToGPU();

        m_ForwardShader = ShaderManager::getInstance().getShader("shaders/attention/add_norm.comp");
        m_BackwardShader = ShaderManager::getInstance().getShader("shaders/attention/backward_add_norm.comp");
        LOG_DEBUG("LayerNorm initialized with shape " + std::to_string(normalizedShape));
    }

    std::shared_ptr<Matrix> LayerNorm::forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& residual) {
        MLGL::Timer timer("LayerNorm::forward");
        int seqLen = input->rows;
        int modelDim = input->cols;
        m_CachedInput = input;
        m_CachedResidual = residual;
        
        // Prepare output: reuse m_CachedOutput if possible
        if (!m_CachedOutput || m_CachedOutput->rows != seqLen || m_CachedOutput->cols != modelDim) {
            m_CachedOutput = std::make_shared<Matrix>(seqLen, modelDim);
            m_CachedOutput->uploadToGPU();
        }

        if (!m_CachedPaddingMask.empty())
            updatePaddingMask(seqLen, m_CachedPaddingMask);


        m_ForwardShader->setUniform("has_padding_mask", !m_CachedPaddingMask.empty());
        if (!m_CachedPaddingMask.empty()) m_ForwardShader->bindBuffer(3, "PaddingMask", m_CachedPaddingMaskBuffer); // Bind padding mask if available
        m_ForwardShader->bindBuffer(0, "InputA", input->buffer);
        m_ForwardShader->bindBuffer(1, "InputB", residual->buffer);
        m_ForwardShader->bindBuffer(2, "Gamma", m_Gamma->buffer);
        m_ForwardShader->bindBuffer(3, "Beta", m_Beta->buffer);
        m_ForwardShader->bindBuffer(4, "Output", m_CachedOutput->buffer);
        m_ForwardShader->setUniform("seq_len", seqLen);
        m_ForwardShader->setUniform("model_dim", modelDim);
        m_ForwardShader->setUniform("epsilon", m_Epsilon);

        // Use correct workgroup count for local_size_x = 32
        int outputWorkgroupsX = (seqLen + 31) / 32;
        m_ForwardShader->dispatch(outputWorkgroupsX, 1, 1);
        for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);

        return m_CachedOutput;
    }

    std::shared_ptr<Matrix> LayerNorm::forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& residual, const std::vector<int>& paddingMask) {
        MLGL::Timer timer("LayerNorm::forward (with mask)");
        // Store padding mask for use in shaders
        m_CachedPaddingMask = paddingMask;

        // Use the existing forward implementation
        return forward(input, residual);
    }

    void LayerNorm::backward(
        const std::shared_ptr<Matrix>& gradOutput,
        const std::shared_ptr<Matrix>& input,
        const std::shared_ptr<Matrix>& residual,
        const GLuint gradMaskBuffer
    ) {
        MLGL::Timer timer("LayerNorm::backward");
        int seqLen = input->rows;
        int modelDim = input->cols;

        // Reuse gradient matrices if possible
        if (!m_GradInput || m_GradInput->rows != seqLen || m_GradInput->cols != modelDim) {
            m_GradInput = std::make_shared<Matrix>(seqLen, modelDim);
            m_GradInput->uploadToGPU();
        }
        if (!m_GradResidual || m_GradResidual->rows != seqLen || m_GradResidual->cols != modelDim) {
            m_GradResidual = std::make_shared<Matrix>(seqLen, modelDim);
            m_GradResidual->uploadToGPU();
        }
        // Allocate grad gamma/beta as (modelDim, seqLen) for per-position accumulation
        if (!m_GradGamma || m_GradGamma->rows != modelDim || m_GradGamma->cols != seqLen) {
            m_GradGamma = std::make_shared<Matrix>(modelDim, seqLen, 0.0f);
            m_GradGamma->uploadToGPU();
        }
        if (!m_GradBeta || m_GradBeta->rows != modelDim || m_GradBeta->cols != seqLen) {
            m_GradBeta = std::make_shared<Matrix>(modelDim, seqLen, 0.0f);
            m_GradBeta->uploadToGPU();
        }

        if (!m_CachedPaddingMask.empty())
            updatePaddingMask(seqLen, m_CachedPaddingMask);

        m_BackwardShader->bindBuffer(0, "GradOutput", gradOutput->buffer);
        m_BackwardShader->bindBuffer(1, "InputA", input->buffer);
        m_BackwardShader->bindBuffer(2, "InputB", residual->buffer);
        m_BackwardShader->bindBuffer(3, "Gamma", m_Gamma->buffer);
        m_BackwardShader->bindBuffer(4, "Beta", m_Beta->buffer);
        m_BackwardShader->bindBuffer(5, "GradInputA", m_GradInput->buffer);
        m_BackwardShader->bindBuffer(6, "GradInputB", m_GradResidual->buffer);
        m_BackwardShader->bindBuffer(7, "GradGamma", m_GradGamma->buffer);
        m_BackwardShader->bindBuffer(8, "GradBeta", m_GradBeta->buffer);
        if(gradMaskBuffer) 
            m_BackwardShader->bindBuffer(9, "GradOutputMask", gradMaskBuffer);
        m_BackwardShader->setUniform("use_mask", gradMaskBuffer != 0);

        m_BackwardShader->setUniform("seq_len", seqLen);
        m_BackwardShader->setUniform("model_dim", modelDim);
        m_BackwardShader->setUniform("epsilon", m_Epsilon);



        // Use correct workgroup count for local_size_x = 32
        int outputWorkgroupsX = (seqLen + 31) / 32;
        m_BackwardShader->dispatch(outputWorkgroupsX, 1, 1);

        // Unbind buffers
        for (int i = 0; i <= 9; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }
    void LayerNorm::updatePaddingMask(int seq_len, const std::vector<int>& mask) {
        if (m_CachedPaddingMaskBuffer == 0) {
            glGenBuffers(1, &m_CachedPaddingMaskBuffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_CachedPaddingMaskBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, seq_len * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_CachedPaddingMaskBuffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, mask.size() * sizeof(int), mask.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
} 
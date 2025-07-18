#include "LayerNorm.h"
#include "Logger.h"
#include <cmath>
#include <algorithm>
#include "ShaderCPUAnalogs.h"

static bool g_LayerNormDebug = false;

namespace NNGL {
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
        NNGL::Timer timer("LayerNorm::forward");
        int seqLen = input->cols;
        int modelDim = input->rows;
        m_CachedInput = input;
        m_CachedResidual = residual;
        
        // Prepare output: reuse m_CachedOutput if possible
        if (!m_CachedOutput || m_CachedOutput->rows != modelDim || m_CachedOutput->cols != seqLen) {
            m_CachedOutput = std::make_shared<Matrix>(modelDim, seqLen);
            m_CachedOutput->uploadToGPU();
        }

        m_ForwardShader->bindBuffer(0, "InputA", DEBUG_VALIDATION(input));
        m_ForwardShader->bindBuffer(1, "InputB", DEBUG_VALIDATION(residual));
        m_ForwardShader->bindBuffer(2, "Gamma", DEBUG_VALIDATION(m_Gamma));
        m_ForwardShader->bindBuffer(3, "Beta", m_Beta->buffer); //ment to be 0
        m_ForwardShader->bindBuffer(4, "Output", m_CachedOutput->buffer);
        m_ForwardShader->setUniform("seq_len", seqLen);
        m_ForwardShader->setUniform("model_dim", modelDim);
        m_ForwardShader->setUniform("epsilon", m_Epsilon);
        // Use correct workgroup count for local_size_x = 32
        int outputWorkgroupsX = (seqLen + 31) / 32;
        m_ForwardShader->dispatch(outputWorkgroupsX, 1, 1);

        // Unbind buffers
        for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);

        DEBUG_VALIDATION(m_CachedOutput);

        return m_CachedOutput;
    }

    void LayerNorm::backward(
        const std::shared_ptr<Matrix>& gradOutput,
        const std::shared_ptr<Matrix>& input,
        const std::shared_ptr<Matrix>& residual
    ) {
        NNGL::Timer timer("LayerNorm::backward");
        int seqLen = input->cols;
        int modelDim = input->rows;
        // Reuse gradient matrices if possible
        if (!m_GradInput || m_GradInput->rows != modelDim || m_GradInput->cols != seqLen) {
            m_GradInput = std::make_shared<Matrix>(modelDim, seqLen);
            m_GradInput->uploadToGPU();
        }
        if (!m_GradResidual || m_GradResidual->rows != modelDim || m_GradResidual->cols != seqLen) {
            m_GradResidual = std::make_shared<Matrix>(modelDim, seqLen);
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

        m_BackwardShader->bindBuffer(0, "GradOutput", DEBUG_VALIDATION(gradOutput));
        m_BackwardShader->bindBuffer(1, "InputA", DEBUG_VALIDATION(input));
        m_BackwardShader->bindBuffer(2, "InputB", DEBUG_VALIDATION(residual));
        m_BackwardShader->bindBuffer(3, "Gamma", DEBUG_VALIDATION(m_Gamma));
        m_BackwardShader->bindBuffer(4, "Beta", m_Beta->buffer); //0
        m_BackwardShader->bindBuffer(5, "GradInputA", m_GradInput->buffer);
        m_BackwardShader->bindBuffer(6, "GradInputB", m_GradResidual->buffer);
        m_BackwardShader->bindBuffer(7, "GradGamma", m_GradGamma->buffer);
        m_BackwardShader->bindBuffer(8, "GradBeta", m_GradBeta->buffer);
        m_BackwardShader->setUniform("seq_len", seqLen);
        m_BackwardShader->setUniform("model_dim", modelDim);
        m_BackwardShader->setUniform("epsilon", m_Epsilon);
        // Use correct workgroup count for local_size_x = 32
        int outputWorkgroupsX = (seqLen + 31) / 32;
        m_BackwardShader->dispatch(outputWorkgroupsX, 1, 1);

        // Unbind buffers
        for (int i = 0; i <= 8; ++i) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        DEBUG_VALIDATION(m_GradInput);
        DEBUG_VALIDATION(m_GradResidual);
        DEBUG_VALIDATION(m_GradGamma);
        DEBUG_VALIDATION(m_GradBeta);
    }
} 
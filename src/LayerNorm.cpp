#include "LayerNorm.h"
#include "Logger.h"
#include <cmath>
#include <algorithm>
#include "ShaderCPUAnalogs.h"

static bool g_LayerNormDebug = false;

namespace NNGL {
    LayerNorm::LayerNorm(int normalizedShape, float epsilon)
        : m_NormalizedShape(normalizedShape), m_Epsilon(epsilon), m_LearningRate(0.001f) {
        m_Gamma = std::make_shared<Matrix>(normalizedShape, 1, 1.0f);
        m_Beta = std::make_shared<Matrix>(normalizedShape, 1, 0.0f);
        m_ForwardShader = ShaderManager::getInstance().getShader("shaders/attention/add_norm.comp");
        m_BackwardShader = ShaderManager::getInstance().getShader("shaders/attention/backward_add_norm.comp");
        LOG_DEBUG("LayerNorm initialized with shape " + std::to_string(normalizedShape));
    }

    std::shared_ptr<Matrix> LayerNorm::forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& residual) {
        NNGL::Timer timer("LayerNorm::forward");
        int seqLen = input->rows;
        int modelDim = input->cols;
        m_CachedInput = input;
        m_CachedResidual = residual;
        
        // Prepare output: reuse m_CachedOutput if possible
        if (!m_CachedOutput || m_CachedOutput->rows != seqLen || m_CachedOutput->cols != modelDim) {
            m_CachedOutput = std::make_shared<Matrix>(seqLen, modelDim);
        }
        input->uploadToGPU();
        residual->uploadToGPU();
        m_Gamma->uploadToGPU();
        m_Beta->uploadToGPU();
        m_CachedOutput->uploadToGPU();
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

        m_CachedOutput->downloadFromGPU();
        // Unbind buffers
        for (int i = 0; i <= 4; ++i) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // Explicit CPU computation and comparison
        if (g_LayerNormDebug) {
            std::vector<float> cpuOutput;
            NNGL::attentionAddNormCPU(
                input->getFlatVec(),
                residual->getFlatVec(),
                m_Gamma->getFlatVec(),
                m_Beta->getFlatVec(),
                cpuOutput,
                seqLen, modelDim, m_Epsilon
            );
            NNGL::compareVectors(cpuOutput, m_CachedOutput->getFlatVec(), 1e-4f, true);
        }

        return m_CachedOutput;
    }

    void LayerNorm::backward(
        const std::shared_ptr<Matrix>& gradOutput,
        const std::shared_ptr<Matrix>& input,
        const std::shared_ptr<Matrix>& residual
    ) {
        NNGL::Timer timer("LayerNorm::backward");
        int seqLen = input->rows;
        int modelDim = input->cols;

        // Reuse gradient matrices if possible
        if (!m_GradInput || m_GradInput->rows != seqLen || m_GradInput->cols != modelDim)
            m_GradInput = std::make_shared<Matrix>(seqLen, modelDim);
        if (!m_GradResidual || m_GradResidual->rows != seqLen || m_GradResidual->cols != modelDim)
            m_GradResidual = std::make_shared<Matrix>(seqLen, modelDim);
        // Allocate grad gamma/beta as (modelDim, seqLen) for per-position accumulation
        if (!m_GradGamma || m_GradGamma->rows != modelDim || m_GradGamma->cols != seqLen)
            m_GradGamma = std::make_shared<Matrix>(modelDim, seqLen, 0.0f);
        if (!m_GradBeta || m_GradBeta->rows != modelDim || m_GradBeta->cols != seqLen)
            m_GradBeta = std::make_shared<Matrix>(modelDim, seqLen, 0.0f);
        // Always zero grad gamma/beta before upload
        m_GradGamma->clear(0.0f);
        m_GradBeta->clear(0.0f);

        gradOutput->uploadToGPU();
        input->uploadToGPU();
        residual->uploadToGPU();
        m_Gamma->uploadToGPU();
        m_Beta->uploadToGPU();
        m_GradInput->uploadToGPU();
        m_GradResidual->uploadToGPU();
        m_GradGamma->uploadToGPU();
        m_GradBeta->uploadToGPU();
        m_BackwardShader->bindBuffer(0, "GradOutput", gradOutput->buffer);
        m_BackwardShader->bindBuffer(1, "InputA", input->buffer);
        m_BackwardShader->bindBuffer(2, "InputB", residual->buffer);
        m_BackwardShader->bindBuffer(3, "Gamma", m_Gamma->buffer);
        m_BackwardShader->bindBuffer(4, "Beta", m_Beta->buffer);
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
        m_GradInput->downloadFromGPU();
        m_GradResidual->downloadFromGPU();
        m_GradGamma->downloadFromGPU();
        m_GradBeta->downloadFromGPU();

        // Unbind buffers
        for (int i = 0; i <= 8; ++i) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // Explicit CPU computation and comparison for backward
        if (g_LayerNormDebug) {
            std::vector<float> cpuGradInput, cpuGradResidual, cpuGradGamma, cpuGradBeta;
            NNGL::attentionBackwardAddNormCPU(
                gradOutput->getFlatVec(),
                input->getFlatVec(),
                residual->getFlatVec(),
                m_Gamma->getFlatVec(),
                cpuGradInput, cpuGradResidual, cpuGradGamma, cpuGradBeta,
                seqLen, modelDim, m_Epsilon
            );
            NNGL::compareVectors(cpuGradInput, m_GradInput->getFlatVec(), 1e-4f, true);
            NNGL::compareVectors(cpuGradResidual, m_GradResidual->getFlatVec(), 1e-4f, true);
            // Sum per-position GPU gradGamma/gradBeta to get final [modelDim] vectors
            std::vector<float> gpuGradGamma(modelDim, 0.0f);
            std::vector<float> gpuGradBeta(modelDim, 0.0f);
            const auto& rawGradGamma = m_GradGamma->getFlatVec();
            const auto& rawGradBeta = m_GradBeta->getFlatVec();

            if ((int)rawGradGamma.size() == modelDim * seqLen) {
                for (int dim = 0; dim < modelDim; ++dim) {
                    for (int seq = 0; seq < seqLen; ++seq) {
                        gpuGradGamma[dim] += rawGradGamma[seq * modelDim + dim];
                        gpuGradBeta[dim]  += rawGradBeta[seq * modelDim + dim];
                    }
                }
            } else if ((int)rawGradGamma.size() == modelDim) {
                gpuGradGamma = rawGradGamma;
                gpuGradBeta = rawGradBeta;
            } else {
                LOG_DEBUG("Unexpected gradGamma/gradBeta buffer size: " + std::to_string(rawGradGamma.size()));
                throw std::runtime_error("Unexpected gradGamma/gradBeta buffer size");
            }
            NNGL::compareVectors(cpuGradGamma, gpuGradGamma, 1e-4f, true);
            NNGL::compareVectors(cpuGradBeta, gpuGradBeta, 1e-4f, true);
        }
    }
} 
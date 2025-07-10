#include "LayerNorm.h"
#include "Logger.h"
#include <cmath>
#include <algorithm>

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
        m_CachedInput = std::make_shared<Matrix>(*input);
        m_CachedResidual = std::make_shared<Matrix>(*residual);
        m_CachedMean = std::make_shared<Matrix>(seqLen, 1);
        m_CachedVariance = std::make_shared<Matrix>(seqLen, 1);
        for (int i = 0; i < seqLen; ++i) {
            float sum = 0.0f;
            for (int d = 0; d < modelDim; ++d) {
                sum += (*input)(i, d) + (*residual)(i, d);
            }
            float m = sum / modelDim;
            (*m_CachedMean)(i, 0) = m;
            float v = 0.0f;
            for (int d = 0; d < modelDim; ++d) {
                float val = (*input)(i, d) + (*residual)(i, d);
                v += (val - m) * (val - m);
            }
            (*m_CachedVariance)(i, 0) = v / modelDim;
        }
        // Prepare output
        m_CachedOutput = std::make_shared<Matrix>(seqLen, modelDim);
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
        m_ForwardShader->dispatch(seqLen, 1, 1);
        m_CachedOutput->downloadFromGPU();
        return m_CachedOutput;
    }

    void LayerNorm::backward(
        const std::shared_ptr<Matrix>& gradOutput,
        const std::shared_ptr<Matrix>& input,
        const std::shared_ptr<Matrix>& residual,
        std::shared_ptr<Matrix>& gradInput,
        std::shared_ptr<Matrix>& gradResidual,
        std::shared_ptr<Matrix>& gradGamma,
        std::shared_ptr<Matrix>& gradBeta
    ) {
        NNGL::Timer timer("LayerNorm::backward");
        int seqLen = input->rows;
        int modelDim = input->cols;

        gradInput = std::make_shared<Matrix>(seqLen, modelDim);
        gradResidual = std::make_shared<Matrix>(seqLen, modelDim);
        gradGamma = std::make_shared<Matrix>(modelDim, 1);
        gradBeta = std::make_shared<Matrix>(modelDim, 1);
        gradOutput->uploadToGPU();
        input->uploadToGPU();
        residual->uploadToGPU();
        m_Gamma->uploadToGPU();
        m_Beta->uploadToGPU();
        m_CachedMean->uploadToGPU();
        m_CachedVariance->uploadToGPU();
        gradInput->uploadToGPU();
        gradResidual->uploadToGPU();
        gradGamma->uploadToGPU();
        gradBeta->uploadToGPU();
        m_BackwardShader->bindBuffer(0, "GradOutput", gradOutput->buffer);
        m_BackwardShader->bindBuffer(1, "InputA", input->buffer);
        m_BackwardShader->bindBuffer(2, "InputB", residual->buffer);
        m_BackwardShader->bindBuffer(3, "Gamma", m_Gamma->buffer);
        m_BackwardShader->bindBuffer(4, "Beta", m_Beta->buffer);
        m_BackwardShader->bindBuffer(5, "CachedMean", m_CachedMean->buffer);
        m_BackwardShader->bindBuffer(6, "CachedVar", m_CachedVariance->buffer);
        m_BackwardShader->bindBuffer(7, "GradInputA", gradInput->buffer);
        m_BackwardShader->bindBuffer(8, "GradInputB", gradResidual->buffer);
        m_BackwardShader->bindBuffer(9, "GradGamma", gradGamma->buffer);
        m_BackwardShader->bindBuffer(10, "GradBeta", gradBeta->buffer);
        m_BackwardShader->setUniform("seq_len", seqLen);
        m_BackwardShader->setUniform("model_dim", modelDim);
        m_BackwardShader->setUniform("epsilon", m_Epsilon);
        m_BackwardShader->dispatch(seqLen, 1, 1);
        gradInput->downloadFromGPU();
        gradResidual->downloadFromGPU();
        gradGamma->downloadFromGPU();
        gradBeta->downloadFromGPU();
    }
} 
#pragma once

#include "Matrix.h"
#include "Shader.h"

#include <memory>

namespace NNGL {
    class LayerNorm {
    public:
        LayerNorm(int normalizedShape, float epsilon = 1e-5f);
        ~LayerNorm() = default;

        std::shared_ptr<Matrix> forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& residual);
        void backward(
            const std::shared_ptr<Matrix>& gradOutput,
            const std::shared_ptr<Matrix>& input,
            const std::shared_ptr<Matrix>& residual,
            std::shared_ptr<Matrix>& gradInput,
            std::shared_ptr<Matrix>& gradResidual,
            std::shared_ptr<Matrix>& gradGamma,
            std::shared_ptr<Matrix>& gradBeta
        );

        void setLearningRate(float lr) { m_LearningRate = lr; }
        std::shared_ptr<Matrix> getGamma() { return m_Gamma; }
        std::shared_ptr<Matrix> getBeta() { return m_Beta; }
        std::shared_ptr<Matrix> getCachedMean() { return m_CachedMean; }
        std::shared_ptr<Matrix> getCachedVar() { return m_CachedVariance; }

    private:
        int m_NormalizedShape;
        float m_Epsilon;
        float m_LearningRate;

        // Learnable parameters
        std::shared_ptr<Matrix> m_Gamma;  // Scale parameter
        std::shared_ptr<Matrix> m_Beta;   // Shift parameter

        // Cached values for backprop
        std::shared_ptr<Matrix> m_CachedInput;
        std::shared_ptr<Matrix> m_CachedResidual;
        std::shared_ptr<Matrix> m_CachedMean;
        std::shared_ptr<Matrix> m_CachedVariance;
        std::shared_ptr<Matrix> m_CachedOutput;

        // GPU shader resources
        std::shared_ptr<Shader> m_ForwardShader;
        std::shared_ptr<Shader> m_BackwardShader;
    };
} 
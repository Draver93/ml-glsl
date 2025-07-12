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
            const std::shared_ptr<Matrix>& residual
        );

        std::shared_ptr<Matrix> getGamma() { return m_Gamma; }
        std::shared_ptr<Matrix> getBeta() { return m_Beta; }
        std::shared_ptr<Matrix> getCachedOutput() { return m_CachedOutput; }
        std::shared_ptr<Matrix> getGradInput() { return m_GradInput; }
        std::shared_ptr<Matrix> getGradResidual() { return m_GradResidual; }
        std::shared_ptr<Matrix> getGradGamma() { return m_GradGamma; }
        std::shared_ptr<Matrix> getGradBeta() { return m_GradBeta; }

    private:
        int m_NormalizedShape;
        float m_Epsilon;

        // Learnable parameters
        std::shared_ptr<Matrix> m_Gamma;  // Scale parameter
        std::shared_ptr<Matrix> m_Beta;   // Shift parameter

        // Cached values for backprop
        std::shared_ptr<Matrix> m_CachedInput;
        std::shared_ptr<Matrix> m_CachedResidual;
        std::shared_ptr<Matrix> m_CachedOutput;

        // Gradient matrices for backprop
        std::shared_ptr<Matrix> m_GradInput;
        std::shared_ptr<Matrix> m_GradResidual;
        std::shared_ptr<Matrix> m_GradGamma;
        std::shared_ptr<Matrix> m_GradBeta;

        // GPU shader resources
        std::shared_ptr<Shader> m_ForwardShader;
        std::shared_ptr<Shader> m_BackwardShader;
    };
} 
#pragma once

#include "Matrix.h"
#include <memory>

namespace NNGL {
    class LayerNorm {
    public:
        LayerNorm(int normalizedShape, float epsilon = 1e-5f);
        ~LayerNorm() = default;

        std::shared_ptr<Matrix> forward(const std::shared_ptr<Matrix>& input);
        std::shared_ptr<Matrix> backward(const std::shared_ptr<Matrix>& gradOutput, float learningRate);

        void setLearningRate(float lr) { m_LearningRate = lr; }

    private:
        int m_NormalizedShape;
        float m_Epsilon;
        float m_LearningRate;

        // Learnable parameters
        std::shared_ptr<Matrix> m_Gamma;  // Scale parameter
        std::shared_ptr<Matrix> m_Beta;   // Shift parameter

        // Cached values for backprop
        std::shared_ptr<Matrix> m_CachedInput;
        std::shared_ptr<Matrix> m_CachedNormalized;
        std::shared_ptr<Matrix> m_CachedMean;
        std::shared_ptr<Matrix> m_CachedVariance;
    };
} 
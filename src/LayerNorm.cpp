#include "LayerNorm.h"
#include "Logger.h"
#include <cmath>
#include <algorithm>

namespace NNGL {
    LayerNorm::LayerNorm(int normalizedShape, float epsilon) 
        : m_NormalizedShape(normalizedShape), m_Epsilon(epsilon), m_LearningRate(0.001f) {
        
        // Initialize learnable parameters
        m_Gamma = std::make_shared<Matrix>(normalizedShape, 1, 1.0f);  // Initialize to 1
        m_Beta = std::make_shared<Matrix>(normalizedShape, 1, 0.0f);   // Initialize to 0
        
        LOG_DEBUG("LayerNorm initialized with shape " + std::to_string(normalizedShape));
    }

    std::shared_ptr<Matrix> LayerNorm::forward(const std::shared_ptr<Matrix>& input) {
        // Cache input for backprop
        m_CachedInput = std::make_shared<Matrix>(*input);
        
        int seqLen = input->rows;
        int batchSize = input->cols;
        
        // Create output matrix
        auto output = std::make_shared<Matrix>(seqLen, batchSize);
        
        // For each position in the sequence
        for (int pos = 0; pos < seqLen; ++pos) {
            // Calculate mean for this position across all features
            float sum = 0.0f;
            for (int batch = 0; batch < batchSize; ++batch) {
                sum += (*input)(pos, batch);
            }
            float mean = sum / batchSize;
            
            // Calculate variance for this position
            float variance = 0.0f;
            for (int batch = 0; batch < batchSize; ++batch) {
                float diff = (*input)(pos, batch) - mean;
                variance += diff * diff;
            }
            variance = variance / batchSize;
            
            // Cache mean and variance for backprop
            if (pos == 0) {
                m_CachedMean = std::make_shared<Matrix>(seqLen, 1);
                m_CachedVariance = std::make_shared<Matrix>(seqLen, 1);
            }
            (*m_CachedMean)(pos, 0) = mean;
            (*m_CachedVariance)(pos, 0) = variance;
            
            // Normalize and apply gamma/beta
            float stdDev = std::sqrt(variance + m_Epsilon);
            for (int batch = 0; batch < batchSize; ++batch) {
                float normalized = ((*input)(pos, batch) - mean) / stdDev;
                float gamma = (*m_Gamma)(pos % m_NormalizedShape, 0);
                float beta = (*m_Beta)(pos % m_NormalizedShape, 0);
                (*output)(pos, batch) = gamma * normalized + beta;
            }
        }
        
        // Cache normalized values for backprop
        m_CachedNormalized = std::make_shared<Matrix>(*output);
        
        return output;
    }

    std::shared_ptr<Matrix> LayerNorm::backward(const std::shared_ptr<Matrix>& gradOutput, float learningRate) {
        int seqLen = gradOutput->rows;
        int batchSize = gradOutput->cols;
        
        // Create gradient input
        auto gradInput = std::make_shared<Matrix>(seqLen, batchSize);
        
        // Gradients for gamma and beta
        auto gradGamma = std::make_shared<Matrix>(m_NormalizedShape, 1, 0.0f);
        auto gradBeta = std::make_shared<Matrix>(m_NormalizedShape, 1, 0.0f);
        
        // For each position in the sequence
        for (int pos = 0; pos < seqLen; ++pos) {
            float mean = (*m_CachedMean)(pos, 0);
            float variance = (*m_CachedVariance)(pos, 0);
            float stdDev = std::sqrt(variance + m_Epsilon);
            
            // Calculate gradients for gamma and beta
            for (int batch = 0; batch < batchSize; ++batch) {
                float normalized = ((*m_CachedNormalized)(pos, batch) - (*m_Beta)(pos % m_NormalizedShape, 0)) / (*m_Gamma)(pos % m_NormalizedShape, 0);
                float gradOut = (*gradOutput)(pos, batch);
                
                // Gradient for gamma
                (*gradGamma)(pos % m_NormalizedShape, 0) += gradOut * normalized;
                
                // Gradient for beta
                (*gradBeta)(pos % m_NormalizedShape, 0) += gradOut;
            }
            
            // Calculate gradient for input
            for (int batch = 0; batch < batchSize; ++batch) {
                float normalized = ((*m_CachedInput)(pos, batch) - mean) / stdDev;
                float gradOut = (*gradOutput)(pos, batch);
                float gamma = (*m_Gamma)(pos % m_NormalizedShape, 0);
                
                // Gradient for input (simplified)
                float gradNorm = gradOut * gamma / stdDev;
                (*gradInput)(pos, batch) = gradNorm;
            }
        }
        
        // Update parameters
        for (int i = 0; i < m_NormalizedShape; ++i) {
            (*m_Gamma)(i, 0) -= learningRate * (*gradGamma)(i, 0);
            (*m_Beta)(i, 0) -= learningRate * (*gradBeta)(i, 0);
        }
        
        return gradInput;
    }
} 
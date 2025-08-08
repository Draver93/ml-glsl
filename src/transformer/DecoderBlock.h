#pragma once

#include "AttentionBlock.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "LayerNorm.h"
#include <memory>

namespace MLGL {

    class DecoderBlock {
    public:
        DecoderBlock(int modelDim, int hiddenDim, int seqLen);
        DecoderBlock(const char* data);
        int getSaveSize();
        const char* save();

        std::shared_ptr<Matrix> forward(
            std::shared_ptr<Matrix> input,
            const std::vector<int>& paddingMask
        );
        
        std::shared_ptr<Matrix> backward(std::shared_ptr<Matrix> gradOutput, const GLuint gradMaskBuffer, float learningRate);
    private:
        std::shared_ptr<Matrix> add(std::shared_ptr<Matrix> dst, std::shared_ptr<Matrix> src);
    private:
        std::unique_ptr<AttentionBlock> m_MaskedSelfAttn;
        std::unique_ptr<NeuralNetwork> m_FeedForward;
        std::unique_ptr<LayerNorm> m_AddNorm1;
        std::unique_ptr<LayerNorm> m_AddNorm2;
        std::shared_ptr<Matrix> m_CachedInput;
        int m_ModelDim, m_SeqLen;
        std::shared_ptr<Shader> m_AddCompute;
    };
}
#include "DecoderBlock.h"
#include "Logger.h"

#include <memory>
#include <iostream>

namespace MLGL {

    DecoderBlock::DecoderBlock(int modelDim, int hiddenDim, int seqLen) : m_ModelDim(modelDim), m_SeqLen(seqLen) {
        m_MaskedSelfAttn = std::make_unique<AttentionBlock>(modelDim, /*numHeads*/64, m_SeqLen, /*isMasked=*/true);

        m_FeedForward = std::make_unique<NeuralNetwork>(m_SeqLen);
        m_FeedForward->addLayer(modelDim, hiddenDim, MLGL::ActivationFnType::RELU);
        m_FeedForward->addLayer(hiddenDim, modelDim, MLGL::ActivationFnType::RELU);

        m_AddNorm1 = std::make_unique<LayerNorm>(modelDim);
        m_AddNorm2 = std::make_unique<LayerNorm>(modelDim);

        m_AddCompute = ShaderManager::getInstance().getShader("shaders/decoder/add_inplace.comp");
    }

    DecoderBlock::DecoderBlock(const char* data) {
        if (!data) throw std::invalid_argument("Data pointer cannot be null");

        const char* ptr = data;

        // Read basic parameters
        std::memcpy(&m_ModelDim, ptr, sizeof(int));
        ptr += sizeof(int);

        // Load AttentionBlock from serialized data
        m_MaskedSelfAttn = std::make_unique<AttentionBlock>(ptr);
        ptr += m_MaskedSelfAttn->getSaveSize();

        // Load NeuralNetwork from serialized data
        m_FeedForward = std::make_unique<NeuralNetwork>(ptr);
        ptr += m_FeedForward->getSaveSize();

        m_AddNorm1 = std::make_unique<LayerNorm>(m_ModelDim);
        m_AddNorm2 = std::make_unique<LayerNorm>(m_ModelDim);

        m_AddCompute = ShaderManager::getInstance().getShader("shaders/decoder/add_inplace.comp");

        LOG_DEBUG("[DECODER LOAD] DecoderBlock loaded successfully from binary buffer");

    }
    int DecoderBlock::getSaveSize() {
        // Basic parameters: modelDim
        size_t basic_params_size = 1 * sizeof(int);

        // Size of each component
        size_t attention_size = m_MaskedSelfAttn->getSaveSize();
        size_t feedforward_size = m_FeedForward->getSaveSize();

        return basic_params_size + attention_size + feedforward_size;
    }

    const char* DecoderBlock::save() {
        // Allocate buffer (caller is responsible for freeing this memory)
        char* buffer = new char[getSaveSize()];
        char* ptr = buffer;

        // Save basic parameters
        std::memcpy(ptr, &m_ModelDim, sizeof(int));
        ptr += sizeof(int);

        // Save AttentionBlock
        const char* attention_data = m_MaskedSelfAttn->save();
        std::memcpy(ptr, attention_data, m_MaskedSelfAttn->getSaveSize());
        ptr += m_MaskedSelfAttn->getSaveSize();
        delete[] attention_data; // Clean up the temporary buffer

        // Save NeuralNetwork
        const char* feedforward_data = m_FeedForward->save();
        std::memcpy(ptr, feedforward_data, m_FeedForward->getSaveSize());
        ptr += m_FeedForward->getSaveSize();
        delete[] feedforward_data; // Clean up the temporary buffer


        LOG_DEBUG("[DECODER SAVE] DecoderBlock saved successfully to binary buffer");

        return buffer;
    }

    std::shared_ptr<Matrix> DecoderBlock::forward(std::shared_ptr<Matrix> input, const std::vector<int>& paddingMask) {
        MLGL::Timer timer("DecoderOnlyBlock::forward");
        m_CachedInput = input;

        // 1. Masked self-attention with residual connection
        auto maskedOut = m_MaskedSelfAttn->forward(input, nullptr, paddingMask);
        auto addNorm1Out = m_AddNorm1->forward(maskedOut, input, paddingMask);

        // 2. Feed-forward network with residual connection
        auto mlpOut = m_FeedForward->forward(addNorm1Out);
        auto addNorm2Out = m_AddNorm2->forward(mlpOut, addNorm1Out, paddingMask);

        return addNorm2Out;
    }

    std::shared_ptr<Matrix> DecoderBlock::backward(std::shared_ptr<Matrix> gradOutput, const GLuint gradMaskBuffer, float learningRate) {
        MLGL::Timer timer("DecoderOnlyBlock::backward");

        // Backprop through addNorm2 (main: mlpOut, residual: addNorm1Out)
        m_AddNorm2->backward(gradOutput, m_FeedForward->getCachedOutput(), m_AddNorm1->getCachedOutput(), gradMaskBuffer);
        auto gradFFNInput = m_FeedForward->backward(m_AddNorm2->getGradInput(), learningRate);
        
        gradFFNInput = add(gradFFNInput, m_AddNorm2->getGradResidual());

        // Backprop through addNorm1 (main: maskedOut, residual: input)
        m_AddNorm1->backward(gradFFNInput, m_MaskedSelfAttn->getCachedOutput(), m_CachedInput);
        auto [gradFromMaskedSelf, maskedGradContext] = m_MaskedSelfAttn->backward(m_AddNorm1->getGradInput(), nullptr, learningRate);
        
        auto result = add(gradFromMaskedSelf, m_AddNorm1->getGradResidual());

        return result;
    }

    std::shared_ptr<Matrix> DecoderBlock::add(std::shared_ptr<Matrix> dst, std::shared_ptr<Matrix> src) {
        MLGL::Timer timer("DecoderBlock::add");

        m_AddCompute->bindBuffer(0, "InOutA", dst->buffer);
        m_AddCompute->bindBuffer(1, "InB", src->buffer);

        if (dst->rows == src->rows && dst->cols == src->cols) m_AddCompute->setUniform("transpose_b", 0);
        else if (dst->rows == src->cols && dst->cols == src->rows) m_AddCompute->setUniform("transpose_b", 1);
        else throw std::invalid_argument("Wrong matrix dims");

        m_AddCompute->setUniform("rowsA", dst->rows);
        m_AddCompute->setUniform("colsA", dst->cols);

        int workgroupsX = (dst->rows + 31) / 32;
        m_AddCompute->dispatch(workgroupsX, 1, 1);
        for (int i = 0; i <= 1; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);

        return dst;
    }
}
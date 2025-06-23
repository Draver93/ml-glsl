#include "AttentionBlock.h"

namespace NNGL {
    AttentionBlock::AttentionBlock(int modelDimensions, int headDimensions, int seqLen, bool mask)
        : m_ModelDim(modelDimensions),  m_HeadDim(headDimensions),
        m_UseMask(mask), m_SeqLen(seqLen) {

        // Weight matrices: [input_dim, head_dim]
        m_WeightQueryMat = std::make_shared<Matrix>(m_ModelDim, m_HeadDim);
        m_WeightQueryMat->randomize();
        m_WeightQueryMat->uploadToGPU();

        m_WeightKeyMat = std::make_shared<Matrix>(m_ModelDim, m_HeadDim);
        m_WeightKeyMat->randomize();
        m_WeightKeyMat->uploadToGPU();

        m_WeightValueMat = std::make_shared<Matrix>(m_ModelDim, m_HeadDim);
        m_WeightValueMat->randomize();
        m_WeightValueMat->uploadToGPU();

        // GradInput: [seq_len, input_dim]
        m_GradQueryInputMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);
        m_GradKeyInputMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);
        m_GradKeyValueMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);

        // GradWeight: [input_dim, head_dim]
        m_GradWeightQueryMat = std::make_shared<Matrix>(m_ModelDim, m_HeadDim, 0);
        m_GradWeightKeyMat = std::make_shared<Matrix>(m_ModelDim, m_HeadDim, 0);
        m_GradWeightValueMat = std::make_shared<Matrix>(m_ModelDim, m_HeadDim, 0);

        int m_NumHeads = 1; // single head
        // Allocate output buffer: [seq_len, m_NumHeads * head_dim]
        m_OutputMat = std::make_shared<Matrix>(seqLen, m_NumHeads * m_HeadDim);

        // Output: [seq_len, head_dim]
        m_OutQueryMat = std::make_shared<Matrix>(m_SeqLen, m_HeadDim);
        m_OutKeyMat = std::make_shared<Matrix>(m_SeqLen, m_HeadDim);
        m_OutValueMat = std::make_shared<Matrix>(m_SeqLen, m_HeadDim);

        m_ForwardPassWeightsCompute = ShaderManager::getInstance().getShader("shaders/self_attention_forward_weights.comp");
        m_ForwardPassScoreCompute = ShaderManager::getInstance().getShader("shaders/self_attention_forward_score.comp");

        m_GradInputCompute = ShaderManager::getInstance().getShader("shaders/self_attention_backward_input.comp");
        m_GradWeightCompute = ShaderManager::getInstance().getShader("shaders/self_attention_backward_weight.comp");

        m_WeightsUpdatePassCompute = ShaderManager::getInstance().getShader("shaders/self_attention_weights_update.comp");
    }

    std::shared_ptr<Matrix> AttentionBlock::forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& input_kv) {
        {
            input->uploadToGPU();
            if(input_kv) input_kv->uploadToGPU();

            m_ForwardPassWeightsCompute->bindBuffer(0, "InputQ", input->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(0, "InputKV", input_kv ? input_kv->buffer : input->buffer);

            m_ForwardPassWeightsCompute->bindBuffer(1, "WeightQ", m_WeightQueryMat->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(2, "WeightK", m_WeightKeyMat->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(3, "WeightV", m_WeightValueMat->buffer);

            m_ForwardPassWeightsCompute->bindBuffer(4, "OutputQ", m_OutQueryMat->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(5, "OutputK", m_OutKeyMat->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(6, "OutputV", m_OutValueMat->buffer);

            m_ForwardPassWeightsCompute->setUniform("head_dim", m_HeadDim);
            m_ForwardPassWeightsCompute->setUniform("input_dim", input->cols);
            m_ForwardPassWeightsCompute->setUniform("seq_len", input->rows);

            int workgroups_x = (input->rows * m_HeadDim + 31) / 32;
            m_ForwardPassWeightsCompute->dispatch(workgroups_x, 1, 1);

            m_OutValueMat->print();
            m_OutValueMat->downloadFromGPU();
            m_OutValueMat->print();
            // Unbind
            for (int i = 0; i <= 6; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
        {
            input->uploadToGPU();

            // Bind buffers
            m_ForwardPassScoreCompute->bindBuffer(0, "BufferQ", m_OutQueryMat->buffer);  // Q
            m_ForwardPassScoreCompute->bindBuffer(1, "BufferK", m_OutKeyMat->buffer);    // K
            m_ForwardPassScoreCompute->bindBuffer(2, "BufferV", m_OutValueMat->buffer);  // V

            m_ForwardPassScoreCompute->bindBuffer(3, "BufferOutput", m_OutputMat->buffer);

            // Set uniforms
            m_ForwardPassScoreCompute->setUniform("seq_len", input->rows);
            m_ForwardPassScoreCompute->setUniform("head_dim", m_HeadDim);
            m_ForwardPassScoreCompute->setUniform("use_mask", m_UseMask ? 1 : 0);
            float inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(m_HeadDim));
            m_ForwardPassScoreCompute->setUniform("inv_sqrt_head_dim", inv_sqrt_head_dim);

            // Dispatch with 2D workgroups: X = seq_len, Y = head_dim (rounded up)
            int workgroups_x = (input->rows + 15) / 16;
            int workgroups_y = (m_HeadDim + 15) / 16;
            m_ForwardPassScoreCompute->dispatch(workgroups_x, workgroups_y, 1);

            // Download results and print
            m_OutputMat->downloadFromGPU();
            m_OutputMat->print();

            // Unbind all buffers
            for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
        return m_OutputMat;
    }
 
    // gradOutput one of concat heads gradients
    void AttentionBlock::backward(const std::shared_ptr<Matrix>& gradOutput, const std::shared_ptr<Matrix>& input) {

        gradOutput->uploadToGPU();
        input->uploadToGPU();

        const int seq_len = input->rows;
        const int input_dim = input->cols;
        const int head_dim = m_HeadDim;

        const int workgroups_input = (seq_len * input_dim + 31) / 32;
        const int workgroups_weight = (input_dim * head_dim + 31) / 32;

        // === Query ===

        // Compute GradInput
        m_GradInputCompute->setUniform("seq_len", seq_len);
        m_GradInputCompute->setUniform("input_dim", input_dim);
        m_GradInputCompute->setUniform("head_dim", head_dim);

        m_GradInputCompute->bindBuffer(0, "GradOutput", gradOutput->buffer);
        m_GradInputCompute->bindBuffer(1, "Weight", m_WeightQueryMat->buffer);
        m_GradInputCompute->bindBuffer(2, "GradInput", m_GradQueryInputMat->buffer);
        m_GradInputCompute->dispatch(workgroups_input, 1, 1);

        // Compute GradWeight
        m_GradWeightCompute->setUniform("seq_len", seq_len);
        m_GradWeightCompute->setUniform("input_dim", input_dim);
        m_GradWeightCompute->setUniform("head_dim", head_dim);

        m_GradWeightCompute->bindBuffer(0, "Input", input->buffer);
        m_GradWeightCompute->bindBuffer(1, "GradOutput", gradOutput->buffer);
        m_GradWeightCompute->bindBuffer(2, "GradWeight", m_GradWeightQueryMat->buffer);
        m_GradWeightCompute->dispatch(workgroups_weight, 1, 1);

        // === Key ===
        m_GradInputCompute->bindBuffer(1, "Weight", m_WeightKeyMat->buffer);
        m_GradInputCompute->bindBuffer(2, "GradInput", m_GradKeyInputMat->buffer);
        m_GradInputCompute->dispatch(workgroups_input, 1, 1);

        m_GradWeightCompute->bindBuffer(2, "GradWeight", m_GradWeightKeyMat->buffer);
        m_GradWeightCompute->dispatch(workgroups_weight, 1, 1);

        // === Value ===
        m_GradInputCompute->bindBuffer(1, "Weight", m_WeightValueMat->buffer);
        m_GradInputCompute->bindBuffer(2, "GradInput", m_GradKeyValueMat->buffer);
        m_GradInputCompute->dispatch(workgroups_input, 1, 1);

        m_GradWeightCompute->bindBuffer(2, "GradWeight", m_GradWeightValueMat->buffer);
        m_GradWeightCompute->dispatch(workgroups_weight, 1, 1);

        // === Update weights ===
        updateWeights(input, m_GradWeightQueryMat, 0.1f);
        updateWeights(input, m_GradWeightKeyMat, 0.1f);
        updateWeights(input, m_GradWeightValueMat, 0.1f);

        // === Cleanup ===
        for (int i = 0; i <= 2; ++i)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }

    void AttentionBlock::updateWeights(const std::shared_ptr<Matrix>& weight, const std::shared_ptr<Matrix>& gradWeight, float learningRate) {

        weight->uploadToGPU();
        gradWeight->uploadToGPU();

        m_WeightsUpdatePassCompute->bindBuffer(0, "Weight", weight->buffer);
        m_WeightsUpdatePassCompute->bindBuffer(1, "GradWeight", gradWeight->buffer);

        m_WeightsUpdatePassCompute->setUniform("input_dim", weight->rows);
        m_WeightsUpdatePassCompute->setUniform("head_dim", weight->cols);
        m_WeightsUpdatePassCompute->setUniform("learning_rate", learningRate);

        int workgroups_x = (weight->rows * weight->cols + 31) / 32;
        m_WeightsUpdatePassCompute->dispatch(workgroups_x, 1, 1);

        for (int i = 0; i <= 1; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }
}

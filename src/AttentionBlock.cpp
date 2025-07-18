#include "AttentionBlock.h"
#include "Logger.h"
#include "ShaderCPUAnalogs.h"
#include <iostream>
#include "Matrix.h"
#include <iomanip>
#include <algorithm>

// Helper to print a 5x5 slice of a matrix
static void printMatrixSlice(const std::string& name, const std::shared_ptr<NNGL::Matrix>& mat) {
    return;
    if (!mat) { std::cout << name << ": nullptr" << std::endl; return; }
    std::cout << "[DEBUG] " << name << " shape=[" << mat->rows << "," << mat->cols << "]\n";
    int max_rows = std::min(5, mat->rows);
    int max_cols = std::min(5, mat->cols);
    // Print first 5 rows
    for (int r = 0; r < max_rows; ++r) {
        std::cout << "  ";
        for (int c = 0; c < max_cols; ++c) {
            std::cout << std::setw(8) << std::setprecision(4) << (*mat)(r, c) << " ";
        }
        if (mat->cols > 5) std::cout << "... ";
        // Print last 5 columns if matrix is wide
        if (mat->cols > 10) {
            for (int c = mat->cols - 5; c < mat->cols; ++c) {
                std::cout << std::setw(8) << std::setprecision(4) << (*mat)(r, c) << " ";
            }
        }
        std::cout << std::endl;
    }
    if (mat->rows > 5) std::cout << "  ..." << std::endl;
    // Print last 5 rows if matrix is tall
    if (mat->rows > 10) {
        for (int r = mat->rows - 5; r < mat->rows; ++r) {
            std::cout << "  ";
            for (int c = 0; c < max_cols; ++c) {
                std::cout << std::setw(8) << std::setprecision(4) << (*mat)(r, c) << " ";
            }
            if (mat->cols > 5) std::cout << "... ";
            if (mat->cols > 10) {
                for (int c = mat->cols - 5; c < mat->cols; ++c) {
                    std::cout << std::setw(8) << std::setprecision(4) << (*mat)(r, c) << " ";
                }
            }
            std::cout << std::endl;
        }
    }
}

namespace NNGL {
    // Static debug flag for validation
    static bool g_AttentionBlockDebug = false;

    AttentionBlock::AttentionBlock(int modelDimensions, int numHeads, int seqLen, bool mask)
        : m_ModelDim(modelDimensions), m_NumHeads(numHeads), m_HeadDim(modelDimensions / numHeads), m_SeqLen(seqLen), m_UseMask(mask), m_ADAM_Timestep(0) {

        // Validate that model dimension is divisible by number of heads
        if (modelDimensions % numHeads != 0) {
            throw std::runtime_error("Model dimension must be divisible by number of heads");
        }

        glGenBuffers(1, &m_PaddingMaskBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_PaddingMaskBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, seqLen * sizeof(int), nullptr, GL_STATIC_DRAW);

        // Initialize weight matrices (standard: [model_dim, model_dim])
        m_WeightQueryMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim);
        m_WeightKeyMat   = std::make_shared<Matrix>(m_ModelDim, m_ModelDim);
        m_WeightValueMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim);
        // Randomly initialize weights (using RELU initializer for consistency)
        for (int r = 0; r < m_ModelDim; r++) {
            for (int c = 0; c < m_ModelDim; c++) {
                m_WeightQueryMat->set(r, c, NNGL::activationFunctions[NNGL::ActivationFnType::RELU].weight_initializer(m_ModelDim, m_ModelDim));
                m_WeightKeyMat->set(r, c, NNGL::activationFunctions[NNGL::ActivationFnType::RELU].weight_initializer(m_ModelDim, m_ModelDim));
                m_WeightValueMat->set(r, c, NNGL::activationFunctions[NNGL::ActivationFnType::RELU].weight_initializer(m_ModelDim, m_ModelDim));
            }
        }
        m_WeightQueryMat->uploadToGPU();
        m_WeightKeyMat->uploadToGPU();
        m_WeightValueMat->uploadToGPU();

        // Initialize ADAM optimization buffers for Q, K, V weights (match weight shapes)
        m_ADAM_M_QueryMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_M_QueryMat->uploadToGPU();
        m_ADAM_V_QueryMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_V_QueryMat->uploadToGPU();
        m_ADAM_M_KeyMat   = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_M_KeyMat->uploadToGPU();
        m_ADAM_V_KeyMat   = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_V_KeyMat->uploadToGPU();
        m_ADAM_M_ValueMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_M_ValueMat->uploadToGPU();
        m_ADAM_V_ValueMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_V_ValueMat->uploadToGPU();

        // Initialize output matrices for Q, K, V projections (standard: [seq_len, model_dim])
        m_OutQueryMat  = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_OutQueryMat->uploadToGPU();
        m_OutKeyMat    = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_OutKeyMat->uploadToGPU();
        m_OutValueMat  = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_OutValueMat->uploadToGPU();

        // Initialize gradient matrices for Q, K, V projections (standard: [seq_len, model_dim])
        m_GradQueryInputMat  = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_GradQueryInputMat->uploadToGPU();
        m_GradKeyInputMat    = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_GradKeyInputMat->uploadToGPU();
        m_GradValueInputMat  = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_GradValueInputMat->uploadToGPU();

        // Cached forward pass values for backprop (standard: [seq_len, model_dim])
        m_CachedInput   = std::make_shared<Matrix>(m_ModelDim, m_SeqLen); // input: [model_dim, seq_len]
        m_CachedInput->uploadToGPU();
        m_CachedContext = std::make_shared<Matrix>(m_ModelDim, m_SeqLen);
        m_CachedContext->uploadToGPU();
        m_CachedQ       = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_CachedQ->uploadToGPU();
        m_CachedK       = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_CachedK->uploadToGPU();
        m_CachedV       = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_CachedV->uploadToGPU();

        m_CachedScores = std::make_shared<Matrix>(m_NumHeads * m_SeqLen, m_SeqLen);
        m_CachedScores->uploadToGPU();
        m_CachedAttentionWeights = std::make_shared<Matrix>(m_NumHeads * m_SeqLen, m_SeqLen);
        m_CachedAttentionWeights->uploadToGPU();

        // Input gradients (standard: [model_dim, seq_len])
        m_GradInput   = std::make_shared<Matrix>(m_ModelDim, m_SeqLen);
        m_GradInput->uploadToGPU();
        m_GradContext = std::make_shared<Matrix>(m_ModelDim, m_SeqLen);
        m_GradContext->uploadToGPU();

        // Weight gradients (standard: [model_dim, model_dim])
        m_GradWeightQueryMat  = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0);
        m_GradWeightQueryMat->uploadToGPU();
        m_GradWeightKeyMat    = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0);
        m_GradWeightKeyMat->uploadToGPU();
        m_GradWeightValueMat  = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0);
        m_GradWeightValueMat->uploadToGPU();

        // Intermediate gradients for backprop chain (standard: [seq_len, model_dim])
        m_GradQ = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_GradQ->uploadToGPU();
        m_GradK = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_GradK->uploadToGPU();
        m_GradV = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_GradV->uploadToGPU();

        m_GradScores = std::make_shared<Matrix>(m_NumHeads * m_SeqLen, m_SeqLen, 0);
        m_GradScores->uploadToGPU();
        m_GradAttentionWeights = std::make_shared<Matrix>(m_NumHeads * m_SeqLen, m_SeqLen, 0);
        m_GradAttentionWeights->uploadToGPU();

        // Output matrix: [model_dim, seq_len] (concatenated heads)
        m_OutputMat = std::make_shared<Matrix>(m_ModelDim, m_SeqLen);
        m_OutputMat->uploadToGPU();

        // Load shaders
        m_ForwardPassWeightsCompute = ShaderManager::getInstance().getShader("shaders/attention/forward_weights.comp");
        m_ForwardPassScoreCompute = ShaderManager::getInstance().getShader("shaders/attention/forward_score.comp");
        m_SoftmaxCompute = ShaderManager::getInstance().getShader("shaders/attention/softmax.comp");
        m_ForwardPassOutCompute = ShaderManager::getInstance().getShader("shaders/attention/forward_output.comp");

        m_BackwardOutputCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_output.comp");
        m_BackwardScoresCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_scores.comp");
        m_WeightsUpdatePassCompute = ShaderManager::getInstance().getShader("shaders/attention/weights_update_adam.comp");
        m_BackwardProjectionsCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_projections.comp");

        m_GradInputCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_grad_input.comp");
        m_GradWeightCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_grad_weight.comp");
        m_AddMatrixShader = ShaderManager::getInstance().getShader("shaders/attention/add_matrix.comp");
    }

    std::shared_ptr<Matrix> AttentionBlock::forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& context) {
        NNGL::Timer timer("AttentionBlock::forward");
        // CACHE INPUT FOR BACKPROP
        input->downloadFromGPU();
        m_CachedInput->copyFrom(input);
        m_CachedInput->uploadToGPU();

        if (context) {
            if (!m_CachedContext) {
                m_CachedContext = std::make_shared<Matrix>(context->rows, context->cols);
            }
            m_CachedContext->copyFrom(context);
        }

        const auto& input_kv = context ? context : input;

        if (!m_PaddingMask.empty()) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_PaddingMaskBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_PaddingMask.size() * sizeof(int), m_PaddingMask.data(), GL_STATIC_DRAW);
        }

        // === STEP 1: Compute Q, K, V projections ===
        {
            input->uploadToGPU();
            input_kv->uploadToGPU();

            m_ForwardPassWeightsCompute->bindBuffer(0, "InputQ", DEBUG_VALIDATION(input));
            m_ForwardPassWeightsCompute->bindBuffer(1, "InputKV", DEBUG_VALIDATION(input_kv));
            m_ForwardPassWeightsCompute->bindBuffer(2, "WeightQ", DEBUG_VALIDATION(m_WeightQueryMat));
            m_ForwardPassWeightsCompute->bindBuffer(3, "WeightK", DEBUG_VALIDATION(m_WeightKeyMat));
            m_ForwardPassWeightsCompute->bindBuffer(4, "WeightV", DEBUG_VALIDATION(m_WeightValueMat));

            // CACHE Q, K, V for backprop
            m_ForwardPassWeightsCompute->bindBuffer(5, "OutputQ", m_CachedQ->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(6, "OutputK", m_CachedK->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(7, "OutputV", m_CachedV->buffer);

            m_ForwardPassWeightsCompute->setUniform("model_dim", m_ModelDim);
            m_ForwardPassWeightsCompute->setUniform("num_heads", m_NumHeads);
            m_ForwardPassWeightsCompute->setUniform("seq_len", input->cols);

            int workgroups_x = (input->cols * m_ModelDim + 31) / 32;
            m_ForwardPassWeightsCompute->dispatch(workgroups_x, 1, 1);

            for (int i = 0; i <= 7; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            DEBUG_VALIDATION(m_CachedQ);
            DEBUG_VALIDATION(m_CachedK);
            DEBUG_VALIDATION(m_CachedV);
        }

        // === STEP 2: Compute attention scores (Q * K^T / sqrt(d_k)) ===
        {
            // This computes scores = Q @ K^T / sqrt(head_dim) for each head
            // CACHE the raw scores BEFORE softmax
            m_ForwardPassScoreCompute->bindBuffer(0, "BufferQ", DEBUG_VALIDATION(m_CachedQ));
            m_ForwardPassScoreCompute->bindBuffer(1, "BufferK", DEBUG_VALIDATION(m_CachedK));
            m_ForwardPassScoreCompute->bindBuffer(2, "RawScores", m_CachedScores->buffer);  // CACHE THIS

            m_ForwardPassScoreCompute->setUniform("has_padding_mask", !m_PaddingMask.empty());
            if (!m_PaddingMask.empty()) m_ForwardPassScoreCompute->bindBuffer(3, "PaddingMask", m_PaddingMaskBuffer); // Bind padding mask if available

            m_ForwardPassScoreCompute->setUniform("seq_len", input->cols);
            m_ForwardPassScoreCompute->setUniform("head_dim", m_HeadDim);
            m_ForwardPassScoreCompute->setUniform("num_heads", m_NumHeads);
            m_ForwardPassScoreCompute->setUniform("use_mask", m_UseMask ? 1 : 0);
            float invSqrtHeadDim = 1.0f / std::sqrt(static_cast<float>(m_HeadDim));
            m_ForwardPassScoreCompute->setUniform("inv_sqrt_head_dim", invSqrtHeadDim);

            int workgroups_x = (input->cols + 15) / 16;
            int workgroups_y = (input->cols + 15) / 16;  // seq_len x seq_len
            m_ForwardPassScoreCompute->dispatch(workgroups_x, workgroups_y, 1);

            for (int i = 0; i <= 3; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            DEBUG_VALIDATION(m_CachedScores);
        }

        // === STEP 3: Apply softmax to get attention weights ===
        {
            // Apply softmax to cached scores, store in cached attention weights
            m_SoftmaxCompute->bindBuffer(0, "Input", DEBUG_VALIDATION(m_CachedScores));
            m_SoftmaxCompute->bindBuffer(1, "Output", m_CachedAttentionWeights->buffer);  // CACHE THIS

            m_SoftmaxCompute->setUniform("has_padding_mask", !m_PaddingMask.empty());
            if (!m_PaddingMask.empty()) m_SoftmaxCompute->bindBuffer(2, "PaddingMask", m_PaddingMaskBuffer); // Bind padding mask if available

            m_SoftmaxCompute->setUniform("seq_len", input->cols);
            m_SoftmaxCompute->setUniform("num_heads", m_NumHeads);
            m_SoftmaxCompute->setUniform("use_mask", m_UseMask ? 1 : 0);

            int total_rows = m_NumHeads * input->cols;
            int workgroups = (total_rows + 15) / 16;
            m_SoftmaxCompute->dispatch(workgroups, 1, 1);

            for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            DEBUG_VALIDATION(m_CachedAttentionWeights);
        }

        // === STEP 4: Compute final output (attention_weights @ V) ===
        {
            // Final matrix multiply: attention_weights @ V -> output
            m_ForwardPassOutCompute->bindBuffer(0, "AttentionWeights", DEBUG_VALIDATION(m_CachedAttentionWeights));
            m_ForwardPassOutCompute->bindBuffer(1, "BufferV", DEBUG_VALIDATION(m_CachedV));
            m_ForwardPassOutCompute->bindBuffer(2, "BufferOutput", m_OutputMat->buffer);

            m_ForwardPassOutCompute->setUniform("seq_len", input->cols);
            m_ForwardPassOutCompute->setUniform("head_dim", m_HeadDim);
            m_ForwardPassOutCompute->setUniform("num_heads", m_NumHeads);

            int workgroups_x = (input->cols + 15) / 16;
            int workgroups_y = (m_ModelDim + 15) / 16;
            m_ForwardPassOutCompute->dispatch(workgroups_x, workgroups_y, 1);
            
            for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            DEBUG_VALIDATION(m_OutputMat);
        }
        //we don't download data we leave them in gpu so no need to worry that array is empty
        return m_OutputMat;
    }

    std::shared_ptr<Matrix> AttentionBlock::forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& input_kv, const std::vector<int>& paddingMask) {
        NNGL::Timer timer("AttentionBlock::forward (with mask)");
        // Store padding mask for use in shaders
        m_PaddingMask = paddingMask;
        
        // Use the existing forward implementation
        return forward(input, input_kv);
    }
 
    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> AttentionBlock::backward( const std::shared_ptr<Matrix>& gradOutput, const std::shared_ptr<Matrix>& context, float learningRate ) {
        NNGL::Timer timer("AttentionBlock::backward");

        if (!m_PaddingMask.empty()) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_PaddingMaskBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_PaddingMask.size() * sizeof(int), m_PaddingMask.data(), GL_STATIC_DRAW);
        }
        // === STEP 1: Backward through final matmul (grad_output -> grad_attention_weights, grad_V) ===
        {
            m_BackwardOutputCompute->bindBuffer(0, "GradOutput", gradOutput->buffer);
            m_BackwardOutputCompute->bindBuffer(1, "CachedV", m_CachedV->buffer);
            m_BackwardOutputCompute->bindBuffer(2, "CachedAttentionWeights", m_CachedAttentionWeights->buffer);
            m_BackwardOutputCompute->bindBuffer(3, "GradAttentionWeights", m_GradAttentionWeights->buffer);
            m_BackwardOutputCompute->bindBuffer(4, "GradV", m_GradV->buffer);
            m_BackwardOutputCompute->setUniform("seq_len", m_SeqLen);
            m_BackwardOutputCompute->setUniform("head_dim", m_HeadDim);
            m_BackwardOutputCompute->setUniform("num_heads", m_NumHeads);
            int workgroups_x = (m_NumHeads * m_SeqLen + 15) / 16;
            int workgroups_y = (m_SeqLen + 15) / 16;
            m_BackwardOutputCompute->dispatch(workgroups_x, workgroups_y, 1);
            for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
        // === STEP 2: Backward through softmax ===
        {
            m_BackwardScoresCompute->bindBuffer(0, "GradAttentionWeights", m_GradAttentionWeights->buffer);
            m_BackwardScoresCompute->bindBuffer(1, "CachedAttentionWeights", m_CachedAttentionWeights->buffer);
            m_BackwardScoresCompute->bindBuffer(2, "GradScores", m_GradScores->buffer);
            m_BackwardScoresCompute->setUniform("has_padding_mask", !m_PaddingMask.empty());
            if (!m_PaddingMask.empty())m_BackwardScoresCompute->bindBuffer(3, "PaddingMask", m_PaddingMaskBuffer);
            m_BackwardScoresCompute->setUniform("seq_len", m_SeqLen);
            m_BackwardScoresCompute->setUniform("num_heads", m_NumHeads);
            m_BackwardScoresCompute->setUniform("use_mask", m_UseMask ? 1 : 0);
            int workgroups_x = (m_NumHeads * m_SeqLen + 15) / 16;
            m_BackwardScoresCompute->dispatch(workgroups_x, 1, 1);

            for (int i = 0; i <= 3; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
        // === STEP 3: Backward through scores computation (Q @ K^T / sqrt(d_k)) ===
        {
            m_BackwardProjectionsCompute->bindBuffer(0, "GradScores", m_GradScores->buffer);
            m_BackwardProjectionsCompute->bindBuffer(1, "CachedQ", m_CachedQ->buffer);
            m_BackwardProjectionsCompute->bindBuffer(2, "CachedK", m_CachedK->buffer);
            m_BackwardProjectionsCompute->bindBuffer(3, "GradQ", m_GradQ->buffer);
            m_BackwardProjectionsCompute->bindBuffer(4, "GradK", m_GradK->buffer);
            m_BackwardProjectionsCompute->setUniform("seq_len", m_SeqLen);
            m_BackwardProjectionsCompute->setUniform("head_dim", m_HeadDim);
            m_BackwardProjectionsCompute->setUniform("num_heads", m_NumHeads);
            float invSqrtHeadDim = 1.0f / std::sqrt(static_cast<float>(m_HeadDim));
            m_BackwardProjectionsCompute->setUniform("inv_sqrt_head_dim", invSqrtHeadDim);
            int workgroups_x = (m_NumHeads * m_SeqLen * m_HeadDim + 31) / 32;
            m_BackwardProjectionsCompute->dispatch(workgroups_x, 1, 1);
            for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
        // === STEP 4: Backward through linear projections ===
        const auto& keyValueInput = context ? m_CachedContext : m_CachedInput;
        computeProjectionGradients(m_GradQ, m_CachedInput, m_WeightQueryMat, m_GradInput, m_GradWeightQueryMat);
        if (context) {
            computeProjectionGradients(m_GradK, keyValueInput, m_WeightKeyMat, m_GradContext, m_GradWeightKeyMat);
        } else {
            auto tempGradInput = std::make_shared<Matrix>(m_ModelDim, m_SeqLen, 0);
            tempGradInput->uploadToGPU();
            computeProjectionGradients(m_GradK, keyValueInput, m_WeightKeyMat, tempGradInput, m_GradWeightKeyMat);
            addMatricesGPU(m_GradInput, tempGradInput, m_GradInput);
        }
        if (context) {
            auto tempGradContext = std::make_shared<Matrix>(m_ModelDim, m_SeqLen, 0);
            tempGradContext->uploadToGPU();
            computeProjectionGradients(m_GradV, keyValueInput, m_WeightValueMat, tempGradContext, m_GradWeightValueMat);
            addMatricesGPU(m_GradContext, tempGradContext, m_GradContext);

        } else {
            auto tempGradInput = std::make_shared<Matrix>(m_ModelDim, m_SeqLen, 0);
            tempGradInput->uploadToGPU();
            computeProjectionGradients(m_GradV, keyValueInput, m_WeightValueMat, tempGradInput, m_GradWeightValueMat);
            addMatricesGPU(m_GradInput, tempGradInput, m_GradInput);

        }
        // === STEP 5: Update weights with ADAM ===
        updateWeights(m_WeightQueryMat, m_GradWeightQueryMat, m_ADAM_M_QueryMat, m_ADAM_V_QueryMat, learningRate);
        updateWeights(m_WeightKeyMat, m_GradWeightKeyMat, m_ADAM_M_KeyMat, m_ADAM_V_KeyMat, learningRate);
        updateWeights(m_WeightValueMat, m_GradWeightValueMat, m_ADAM_M_ValueMat, m_ADAM_V_ValueMat, learningRate);
        m_ADAM_Timestep++;
        return std::make_pair(m_GradInput, m_GradContext);
    }

    void AttentionBlock::updateWeights(const std::shared_ptr<Matrix>& weight, const std::shared_ptr<Matrix>& gradWeight, 
                                     const std::shared_ptr<Matrix>& adamM, const std::shared_ptr<Matrix>& adamV, float learningRate) {

        m_WeightsUpdatePassCompute->bindBuffer(0, "Weight", weight->buffer);
        m_WeightsUpdatePassCompute->bindBuffer(1, "GradWeight", gradWeight->buffer);
        m_WeightsUpdatePassCompute->bindBuffer(2, "ADAM_M", adamM->buffer);
        m_WeightsUpdatePassCompute->bindBuffer(3, "ADAM_V", adamV->buffer);

        m_WeightsUpdatePassCompute->setUniform("model_dim", weight->rows);
        m_WeightsUpdatePassCompute->setUniform("learning_rate", learningRate);
        m_WeightsUpdatePassCompute->setUniform("ADAM_beta1", 0.9f);
        m_WeightsUpdatePassCompute->setUniform("ADAM_beta2", 0.999f);
        m_WeightsUpdatePassCompute->setUniform("ADAM_timestep", m_ADAM_Timestep);

        int workgroups_x = (weight->rows * weight->cols + 31) / 32;
        m_WeightsUpdatePassCompute->dispatch(workgroups_x, 1, 1);

        for (int i = 0; i <= 3; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }

    void AttentionBlock::computeProjectionGradients(const std::shared_ptr<Matrix>& gradProjection, const std::shared_ptr<Matrix>& cachedInput, const std::shared_ptr<Matrix>& weight,
                                                                                                                                                std::shared_ptr<Matrix>& gradInput, std::shared_ptr<Matrix>& gradWeight) {
        // grad_input = grad_projection @ W^T
        // grad_weight = input^T @ grad_projection
        {
            m_GradInputCompute->bindBuffer(0, "GradProjection", gradProjection->buffer);
            m_GradInputCompute->bindBuffer(1, "Weight", weight->buffer);
            m_GradInputCompute->bindBuffer(2, "GradInput", gradInput->buffer);

            m_GradInputCompute->setUniform("seq_len", m_SeqLen);
            m_GradInputCompute->setUniform("model_dim", m_ModelDim);

            // Dispatch with appropriate work group sizes
            int workgroups_x = (m_ModelDim + 15) / 16;
            int workgroups_y = (m_SeqLen + 15) / 16;
            m_GradInputCompute->dispatch(workgroups_x, workgroups_y, 1);
            
            for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === Compute grad_weight = input^T @ grad_projection ===
        {
            m_GradWeightCompute->bindBuffer(0, "CachedInput", cachedInput->buffer);
            m_GradWeightCompute->bindBuffer(1, "GradProjection", gradProjection->buffer);
            m_GradWeightCompute->bindBuffer(2, "GradWeight", gradWeight->buffer);

            m_GradWeightCompute->setUniform("seq_len", m_SeqLen);
            m_GradWeightCompute->setUniform("model_dim", m_ModelDim);

            // Debug: print buffer size and dispatch dimensions
            int workgroups_x = (m_ModelDim + 15) / 16;
            int workgroups_y = (m_ModelDim + 15) / 16; // Changed from m_HeadDim to m_ModelDim

            // Dispatch with appropriate work group sizes
            m_GradWeightCompute->dispatch(workgroups_x, workgroups_y, 1);
            
            for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
    }

    void AttentionBlock::addMatricesGPU(const std::shared_ptr<Matrix>& A, const std::shared_ptr<Matrix>& B, std::shared_ptr<Matrix>& out) {
        // Assumes all matrices are the same shape
        if (!m_AddMatrixShader) {
            m_AddMatrixShader = ShaderManager::getInstance().getShader("shaders/attention/add_matrix.comp");
        }
        int rows = A->rows;
        int cols = A->cols;
        if (B->rows != rows || B->cols != cols || out->rows != rows || out->cols != cols) {
            throw std::runtime_error("Matrix shape mismatch in addMatricesGPU");
        }

        m_AddMatrixShader->bindBuffer(0, "A", A->buffer);
        m_AddMatrixShader->bindBuffer(1, "B", B->buffer);
        m_AddMatrixShader->bindBuffer(2, "Out", out->buffer);
        m_AddMatrixShader->setUniform("rows", rows);
        m_AddMatrixShader->setUniform("cols", cols);
        int workgroups_x = (rows + 15) / 16;
        int workgroups_y = (cols + 15) / 16;
        m_AddMatrixShader->dispatch(workgroups_x, workgroups_y, 1);


        for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }
}

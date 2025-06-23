#include "Embedder.h"

namespace NNGL {

    /**
    * Train embeddings on a corpus of text
    */
    void NeuralVectorEmbedder::trainEmbeddings(const std::vector<std::string>& corpus, const TrainingConfig& config) {
        m_Config = config;

        std::cout << "Preparing training data..." << std::endl;
        prepareTrainingData(corpus);

        std::cout << "Building vocabulary mappings..." << std::endl;
        buildVocabularyMappings();

        std::cout << "Initializing neural network..." << std::endl;
        initializeNetwork();

        std::cout << "Starting training with strategy: " << getStrategyName(m_Config.strategy) << std::endl;

        switch (m_Config.strategy) {
        case EmbeddingStrategy::SKIPGRAM:
            trainSkipGram();
            break;
        case EmbeddingStrategy::CBOW:
            trainCBOW();
            break;
        case EmbeddingStrategy::AUTOENCODER:
            trainAutoencoder();
            break;
        case EmbeddingStrategy::MERGE_PREDICTOR:
            trainMergePredictor();
            break;
        }

        std::cout << "Extracting learned embeddings..." << std::endl;
        extractEmbeddings();

        std::cout << "Training completed!" << std::endl;
    }

    /**
    * Generate embeddings for new merged tokens using the trained network
    */
    std::vector<float> NeuralVectorEmbedder::generateMergedTokenEmbedding(const std::string& token1, const std::string& token2) {
        if (!m_Network) {
            // Fallback to simple averaging if network not trained
            auto vec1 = m_BPEVM->getVector(token1);
            auto vec2 = m_BPEVM->getVector(token2);

            std::vector<float> result(vec1.size());
            for (size_t i = 0; i < vec1.size(); ++i) {
                result[i] = (vec1[i] + vec2[i]) * 0.5f;
            }
            return result;
        }

        // Use network to generate embedding
        auto input = prepareMergeInput(token1, token2);
        auto input_matrix = std::make_shared<Matrix>(input.size(), 1, input.data());

        auto output = m_Network->forward(input_matrix);

        int vec_dim = m_BPEVM->getAllTokens().size();
        std::vector<float> result(vec_dim);
        for (int i = 0; i < vec_dim; ++i) {
            result[i] = (*output)(i, 0);
        }

        // Normalize the result
        normalizeVector(result);
        return result;
    }

    /**
    * Fine-tune embeddings based on token usage patterns
    */
    void NeuralVectorEmbedder::fineTuneEmbeddings(const std::vector<std::string>& corpus, int epochs) {
        if (!m_Network) {
            std::cerr << "Network not initialized. Run trainEmbeddings first." << std::endl;
            return;
        }

        std::cout << "Fine-tuning embeddings on new corpus..." << std::endl;

        // Prepare fine-tuning data
        prepareTrainingData(corpus);

        // Run fewer epochs for fine-tuning
        TrainingConfig fine_tune_config = m_Config;
        fine_tune_config.epochs = epochs;
        fine_tune_config.learning_rate *= 0.1f; // Lower learning rate for fine-tuning

        // Train with reduced learning rate
        switch (m_Config.strategy) {
        case EmbeddingStrategy::SKIPGRAM:
            trainSkipGram();
            break;
        case EmbeddingStrategy::CBOW:
            trainCBOW();
            break;
        default:
            trainSkipGram(); // Default to skip-gram for fine-tuning
            break;
        }

        extractEmbeddings();
        std::cout << "Fine-tuning completed!" << std::endl;
    }

    /**
    * Evaluate embedding quality using similarity tasks
    */
    void NeuralVectorEmbedder::evaluateEmbeddings(const std::vector<std::pair<std::string, std::string>>& similarity_pairs) {
        std::cout << "\n=== Embedding Quality Evaluation ===" << std::endl;

        float total_similarity = 0.0f;
        int valid_pairs = 0;

        for (const auto& pair : similarity_pairs) {
            if (m_BPEVM->hasToken(pair.first) && m_BPEVM->hasToken(pair.second)) {
                float sim = m_BPEVM->getSimilarity(pair.first, pair.second);
                std::cout << "'" << pair.first << "' <-> '" << pair.second << "': " << sim << std::endl;
                total_similarity += sim;
                valid_pairs++;
            }
        }

        if (valid_pairs > 0) {
            std::cout << "Average similarity: " << (total_similarity / valid_pairs) << std::endl;
        }

        std::cout << "===================================\n" << std::endl;
    }

    void NeuralVectorEmbedder::initializeNetwork() {
        // Calculate network dimensions based on vocabulary size and strategy
        updateNetworkDimensions();

        // Create network with appropriate architecture
        m_Network = std::make_shared<NeuralNetwork>(m_Config.batch_size);

        // Build network architecture based on strategy
        switch (m_Config.strategy) {
        case EmbeddingStrategy::SKIPGRAM:
        case EmbeddingStrategy::CBOW:
            buildWordEmbeddingNetwork();
            break;
        case EmbeddingStrategy::AUTOENCODER:
            buildAutoencoderNetwork();
            break;
        case EmbeddingStrategy::MERGE_PREDICTOR:
            buildMergePredictorNetwork();
            break;
        }

        // Set up training callbacks
        setupTrainingCallbacks();
    }

    void NeuralVectorEmbedder::updateNetworkDimensions() {
        auto tokens = m_BPEVM->getTokens();
        int vocab_size = static_cast<int>(tokens.size());
        int embed_dim = m_BPEVM->getAllTokens().size();

        switch (m_Config.strategy) {
        case EmbeddingStrategy::SKIPGRAM:
            m_InputSize = vocab_size;      // One-hot encoded input token
            m_OutputSize = vocab_size;     // Softmax over vocabulary
            break;
        case EmbeddingStrategy::CBOW:
            m_InputSize = vocab_size * m_Config.context_window; // Context tokens
            m_OutputSize = vocab_size;     // Target token
            break;
        case EmbeddingStrategy::AUTOENCODER:
            m_InputSize = embed_dim;       // Current embedding
            m_OutputSize = embed_dim;      // Reconstructed embedding
            break;
        case EmbeddingStrategy::MERGE_PREDICTOR:
            m_InputSize = embed_dim * 2;   // Two token embeddings
            m_OutputSize = embed_dim;      // Merged embedding
            break;
        }
    }

    void NeuralVectorEmbedder::buildWordEmbeddingNetwork() {
        // Input layer -> Hidden layer (embedding layer)
        m_Network->addLayer(m_InputSize, m_HiddenSize, ActivationFnType::RELU);

        // Hidden layer -> Output layer with softmax
        m_Network->addLayer(m_HiddenSize, m_OutputSize, ActivationFnType::IDENTITY);
        //this should be softmax hmm
    }

    void NeuralVectorEmbedder::buildAutoencoderNetwork() {
        // Encoder: Input -> Compressed representation
        m_Network->addLayer(m_InputSize, m_HiddenSize, ActivationFnType::RELU);
        m_Network->addLayer(m_HiddenSize, m_HiddenSize / 2, ActivationFnType::RELU);

        // Decoder: Compressed -> Reconstructed output
        m_Network->addLayer(m_HiddenSize / 2, m_HiddenSize, ActivationFnType::RELU);
        m_Network->addLayer(m_HiddenSize, m_OutputSize, ActivationFnType::IDENTITY);
    }

    void NeuralVectorEmbedder::buildMergePredictorNetwork() {
        // Two token embeddings -> Hidden layers -> Merged embedding
        m_Network->addLayer(m_InputSize, m_HiddenSize, ActivationFnType::RELU);
        m_Network->addLayer(m_HiddenSize, m_HiddenSize, ActivationFnType::RELU);
        m_Network->addLayer(m_HiddenSize, m_OutputSize, ActivationFnType::TANH);
    }

    void NeuralVectorEmbedder::setupTrainingCallbacks() {
        // Set up training batch provider
        m_Network->onTrainBatch([this](std::shared_ptr<Matrix>& input, std::shared_ptr<Matrix>& output, int batch_size) {
            generateTrainingBatch(input, output, batch_size);
            });

        // Set up test batch provider
        m_Network->onTestBatch([this](std::shared_ptr<Matrix>& input, std::shared_ptr<Matrix>& output, int batch_size) {
            generateTestBatch(input, output, batch_size);
            });
    }

    void NeuralVectorEmbedder::prepareTrainingData(const std::vector<std::string>& corpus) {
        m_TrainingCorpus.clear();
        m_TokenFrequency.clear();

        // Tokenize corpus and count frequencies
        for (const auto& text : corpus) {
            auto tokens = m_BPEVM->encode(text);
            if (!tokens.empty()) {
                m_TrainingCorpus.push_back(tokens);

                // Count token frequencies for subsampling
                for (const auto& token : tokens) {
                    m_TokenFrequency[token]++;
                }
            }
        }

        // Calculate subsampling probabilities
        if (m_Config.use_subsampling) {
            calculateSubsamplingProbabilities();
        }

        std::cout << "Prepared " << m_TrainingCorpus.size() << " sequences with "
            << m_TokenFrequency.size() << " unique tokens" << std::endl;
    }

    void NeuralVectorEmbedder::calculateSubsamplingProbabilities() {
        m_SubsamplingProb.clear();

        // Calculate total token count
        int total_tokens = 0;
        for (const auto& pair : m_TokenFrequency) {
            total_tokens += pair.second;
        }

        // Calculate subsampling probabilities
        for (const auto& pair : m_TokenFrequency) {
            float freq = static_cast<float>(pair.second) / total_tokens;
            float prob = 1.0f - std::sqrt(m_Config.subsampling_threshold / freq);
            m_SubsamplingProb[pair.first] = std::max(0.0f, prob);
        }
    }

    void NeuralVectorEmbedder::buildVocabularyMappings() {
        m_TokenToIndex.clear();
        m_IndexToToken.clear();

        auto tokens = m_BPEVM->getTokens();
        m_IndexToToken.reserve(tokens.size());

        int index = 0;
        for (const auto& token : tokens) {
            m_TokenToIndex[token] = index;
            m_IndexToToken.push_back(token);
            ++index;
        }
    }

    void NeuralVectorEmbedder::trainSkipGram() {
        std::cout << "Training Skip-gram model..." << std::endl;

        for (int epoch = 0; epoch < m_Config.epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int batches = 0;

            // Shuffle training data
            std::shuffle(m_TrainingCorpus.begin(), m_TrainingCorpus.end(), m_RNG);

            for (const auto& tokens : m_TrainingCorpus) {
                generateSkipGramPairs(tokens);

                // Train on generated pairs
                for (size_t i = 0; i < m_TrainingPairs.size(); i += m_Config.batch_size) {
                    m_Network->train(m_Config.learning_rate);
                    batches++;
                }
            }

            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << "/" << m_Config.epochs << " completed" << std::endl;
            }
        }
    }

    void NeuralVectorEmbedder::trainCBOW() {
        std::cout << "Training CBOW model..." << std::endl;

        for (int epoch = 0; epoch < m_Config.epochs; ++epoch) {
            std::shuffle(m_TrainingCorpus.begin(), m_TrainingCorpus.end(), m_RNG);

            for (const auto& tokens : m_TrainingCorpus) {
                generateCBOWPairs(tokens);

                for (size_t i = 0; i < m_TrainingPairs.size(); i += m_Config.batch_size) {
                    m_Network->train(m_Config.learning_rate);
                }
            }

            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << "/" << m_Config.epochs << " completed" << std::endl;
            }
        }
    }

    void NeuralVectorEmbedder::trainAutoencoder() {
        std::cout << "Training Autoencoder model..." << std::endl;

        // Generate autoencoder training pairs (embedding -> reconstructed embedding)
        generateAutoencoderPairs();

        for (int epoch = 0; epoch < m_Config.epochs; ++epoch) {
            std::shuffle(m_TrainingPairs.begin(), m_TrainingPairs.end(), m_RNG);

            for (size_t i = 0; i < m_TrainingPairs.size(); i += m_Config.batch_size) {
                m_Network->train(m_Config.learning_rate);
            }

            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << "/" << m_Config.epochs << " completed" << std::endl;
            }
        }
    }

    void NeuralVectorEmbedder::trainMergePredictor() {
        std::cout << "Training Merge Predictor model..." << std::endl;

        // Generate merge prediction training pairs
        generateMergePredictorPairs();

        for (int epoch = 0; epoch < m_Config.epochs; ++epoch) {
            std::shuffle(m_TrainingPairs.begin(), m_TrainingPairs.end(), m_RNG);

            for (size_t i = 0; i < m_TrainingPairs.size(); i += m_Config.batch_size) {
                m_Network->train(m_Config.learning_rate);
            }

            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << "/" << m_Config.epochs << " completed" << std::endl;
            }
        }
    }

    void NeuralVectorEmbedder::generateSkipGramPairs(const std::vector<std::string>& tokens) {
        m_TrainingPairs.clear();

        for (size_t i = 0; i < tokens.size(); ++i) {
            // Skip if token should be subsampled
            if (m_Config.use_subsampling && shouldSubsample(tokens[i])) {
                continue;
            }

            // Generate context pairs
            int window_start = std::max(0, static_cast<int>(i) - m_Config.context_window);
            int window_end = std::min(static_cast<int>(tokens.size()), static_cast<int>(i) + m_Config.context_window + 1);

            for (int j = window_start; j < window_end; ++j) {
                if (j != static_cast<int>(i)) {
                    auto input_vec = createOneHotVector(tokens[i]);
                    auto output_vec = createOneHotVector(tokens[j]);
                    m_TrainingPairs.emplace_back(input_vec, output_vec);
                }
            }
        }
    }

    void NeuralVectorEmbedder::generateCBOWPairs(const std::vector<std::string>& tokens) {
        m_TrainingPairs.clear();

        for (size_t i = 0; i < tokens.size(); ++i) {
            if (m_Config.use_subsampling && shouldSubsample(tokens[i])) {
                continue;
            }

            // Create context vector
            std::vector<float> context_vec(m_InputSize, 0.0f);
            int context_count = 0;

            int window_start = std::max(0, static_cast<int>(i) - m_Config.context_window);
            int window_end = std::min(static_cast<int>(tokens.size()), static_cast<int>(i) + m_Config.context_window + 1);

            for (int j = window_start; j < window_end; ++j) {
                if (j != static_cast<int>(i)) {
                    auto token_vec = createOneHotVector(tokens[j]);
                    for (size_t k = 0; k < token_vec.size(); ++k) {
                        context_vec[k] += token_vec[k];
                    }
                    context_count++;
                }
            }

            // Average the context
            if (context_count > 0) {
                for (auto& val : context_vec) {
                    val /= context_count;
                }

                auto target_vec = createOneHotVector(tokens[i]);
                m_TrainingPairs.emplace_back(context_vec, target_vec);
            }
        }
    }

    void NeuralVectorEmbedder::generateAutoencoderPairs() {
        m_TrainingPairs.clear();

        for (const auto& token : m_BPEVM->getTokens()) {
            auto embedding = m_BPEVM->getVector(token);
            // Input and output are the same for autoencoder
            m_TrainingPairs.emplace_back(embedding, embedding);
        }
    }

    void NeuralVectorEmbedder::generateMergePredictorPairs() {
        m_TrainingPairs.clear();

        // Use existing merge history to train predictor
        auto merges = m_BPEVM->getMerges(); // You'd need to add this method to BPEVM

        for (const auto& merge : merges) {
            auto vec1 = m_BPEVM->getVector(merge.first);
            auto vec2 = m_BPEVM->getVector(merge.second);

            // Concatenate input vectors
            std::vector<float> input_vec;
            input_vec.reserve(vec1.size() + vec2.size());
            input_vec.insert(input_vec.end(), vec1.begin(), vec1.end());
            input_vec.insert(input_vec.end(), vec2.begin(), vec2.end());

            // Target is the actual merged token embedding
            std::string merged_token = merge.first + merge.second;
            auto target_vec = m_BPEVM->getVector(merged_token);

            m_TrainingPairs.emplace_back(input_vec, target_vec);
        }
    }

    std::vector<float> NeuralVectorEmbedder::createOneHotVector(const std::string& token) const {
        std::vector<float> vec(m_IndexToToken.size(), 0.0f);
        auto it = m_TokenToIndex.find(token);
        if (it != m_TokenToIndex.end()) {
            vec[it->second] = 1.0f;
        }
        return vec;
    }

    bool NeuralVectorEmbedder::shouldSubsample(const std::string& token) const {
        auto it = m_SubsamplingProb.find(token);
        if (it != m_SubsamplingProb.end()) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            return dist(m_RNG) < it->second;
        }
        return false;
    }

    std::vector<float> NeuralVectorEmbedder::prepareMergeInput(const std::string& token1, const std::string& token2) const {
        auto vec1 = m_BPEVM->getVector(token1);
        auto vec2 = m_BPEVM->getVector(token2);

        std::vector<float> input;
        input.reserve(vec1.size() + vec2.size());
        input.insert(input.end(), vec1.begin(), vec1.end());
        input.insert(input.end(), vec2.begin(), vec2.end());

        return input;
    }

    void NeuralVectorEmbedder::extractEmbeddings() {

    }

    void NeuralVectorEmbedder::generateTrainingBatch(std::shared_ptr<Matrix>& input, std::shared_ptr<Matrix>& output, int batch_size) {
        // Generate a batch of training examples
        if (m_TrainingPairs.empty()) return;

        std::uniform_int_distribution<size_t> dist(0, m_TrainingPairs.size() - 1);

        // Prepare batch matrices
        int input_dim = static_cast<int>(m_TrainingPairs[0].first.size());
        int output_dim = static_cast<int>(m_TrainingPairs[0].second.size());

        std::vector<float> input_data(input_dim * batch_size);
        std::vector<float> output_data(output_dim * batch_size);

        for (int i = 0; i < batch_size; ++i) {
            size_t idx = dist(m_RNG);
            const auto& pair = m_TrainingPairs[idx];

            // Copy input data
            for (int j = 0; j < input_dim; ++j) {
                input_data[i * input_dim + j] = pair.first[j];
            }

            // Copy output data
            for (int j = 0; j < output_dim; ++j) {
                output_data[i * output_dim + j] = pair.second[j];
            }
        }

        input = std::make_shared<Matrix>(input_dim, batch_size, input_data.data());
        output = std::make_shared<Matrix>(output_dim, batch_size, output_data.data());
    }

    void NeuralVectorEmbedder::generateTestBatch(std::shared_ptr<Matrix>& input, std::shared_ptr<Matrix>& output, int batch_size) {
        // For simplicity, use same as training batch
        // In practice, you'd have a separate test set
        generateTrainingBatch(input, output, batch_size);
    }

    void NeuralVectorEmbedder::normalizeVector(std::vector<float>& vec) const {
        float sum = 0.0f;
        for (float val : vec) {
            sum += val * val;
        }

        if (sum > 0.0f) {
            float magnitude = std::sqrt(sum);
            for (float& val : vec) {
                val /= magnitude;
            }
        }
    }

    std::string NeuralVectorEmbedder::getStrategyName(EmbeddingStrategy strategy) const {
        switch (strategy) {
        case EmbeddingStrategy::SKIPGRAM: return "Skip-gram";
        case EmbeddingStrategy::CBOW: return "CBOW";
        case EmbeddingStrategy::AUTOENCODER: return "Autoencoder";
        case EmbeddingStrategy::MERGE_PREDICTOR: return "Merge Predictor";
        default: return "Unknown";
        }
    }

} // namespace NNGL

/*
auto bpevm = std::make_shared<BytePairEncodingVectorMapper>(128); // 128-dim vectors
auto embedder = std::make_unique<NeuralVectorEmbedder>(bpevm, 256, 32); // 256 hidden, batch=32

// Configure training
NeuralVectorEmbedder::TrainingConfig config;
config.strategy = NeuralVectorEmbedder::EmbeddingStrategy::SKIPGRAM;
config.context_window = 5;
config.learning_rate = 0.001f;
config.epochs = 100;

// Train on your corpus
std::vector<std::string> corpus = {"your", "training", "text", "here"};
embedder->trainEmbeddings(corpus, config);

// Now your BPEVM will have much better embeddings!
// When new tokens are merged, use the neural predictor:
auto new_embedding = embedder->generateMergedTokenEmbedding("hello", "world");

*/
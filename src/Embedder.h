#pragma once

#include "NeuralNetwork.h"
#include "BPEVM.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <iostream>

namespace NNGL {
    /**
 * Neural Vector Embedder - Uses neural networks to learn better token embeddings
 * This class integrates with your existing NeuralNetwork and BPEVM classes
 */
    class NeuralVectorEmbedder {
    public:
        enum class EmbeddingStrategy {
            SKIPGRAM,           // Skip-gram like training (predict context from token)
            CBOW,               // Continuous bag of words (predict token from context)
            AUTOENCODER,        // Autoencoder for dimensionality reduction/feature learning
            MERGE_PREDICTOR     // Predict which tokens should be merged together
        };

        struct TrainingConfig {
            EmbeddingStrategy strategy = EmbeddingStrategy::SKIPGRAM;
            int context_window = 5;
            int negative_samples = 5;
            float learning_rate = 0.001f;
            int epochs = 100;
            int batch_size = 32;
            bool use_subsampling = true;
            float subsampling_threshold = 1e-5f;
        };

    private:
        std::shared_ptr<NeuralNetwork> m_Network;
        std::shared_ptr<BytePairEncodingVectorMapper> m_BPEVM;
        TrainingConfig m_Config;

        // Token frequency for subsampling
        std::unordered_map<std::string, int> m_TokenFrequency;
        std::unordered_map<std::string, float> m_SubsamplingProb;

        // Token to index mapping for efficient processing
        std::unordered_map<std::string, int> m_TokenToIndex;
        std::vector<std::string> m_IndexToToken;

        // Training data structures
        std::vector<std::vector<std::string>> m_TrainingCorpus;
        std::vector<std::pair<std::vector<float>, std::vector<float>>> m_TrainingPairs;

        // Random number generation
        mutable std::mt19937 m_RNG;

        // Network architecture sizes
        int m_InputSize;
        int m_HiddenSize;
        int m_OutputSize;

    public:
        NeuralVectorEmbedder(std::shared_ptr<BytePairEncodingVectorMapper> bpevm,
            int hidden_size = 128,
            int batch_size = 32)
            : m_BPEVM(bpevm)
            , m_Config()
            , m_RNG(std::random_device{}())
            , m_HiddenSize(hidden_size)
        {
            m_Config.batch_size = batch_size;
            initializeNetwork();
        }

        ~NeuralVectorEmbedder() = default;

        /**
         * Train embeddings on a corpus of text
         */
        void trainEmbeddings(const std::vector<std::string>& corpus, const TrainingConfig& config = TrainingConfig{});

        /**
         * Generate embeddings for new merged tokens using the trained network
         */
        std::vector<float> generateMergedTokenEmbedding(const std::string& token1, const std::string& token2);
        /**
         * Fine-tune embeddings based on token usage patterns
         */
        void fineTuneEmbeddings(const std::vector<std::string>& corpus, int epochs = 10);

        /**
         * Evaluate embedding quality using similarity tasks
         */
        void evaluateEmbeddings(const std::vector<std::pair<std::string, std::string>>& similarity_pairs);

        void setTrainingConfig(const TrainingConfig& config) { m_Config = config; }
        const TrainingConfig& getTrainingConfig() const { return m_Config; }

    private:
        void initializeNetwork();

        void updateNetworkDimensions();

        void buildWordEmbeddingNetwork(); 

        void buildAutoencoderNetwork();

        void buildMergePredictorNetwork();

        void setupTrainingCallbacks();

        void prepareTrainingData(const std::vector<std::string>& corpus);

        void calculateSubsamplingProbabilities();

        void buildVocabularyMappings();

        void trainSkipGram();

        void trainCBOW();

        void trainAutoencoder();

        void trainMergePredictor();

        void generateSkipGramPairs(const std::vector<std::string>& tokens);

        void generateCBOWPairs(const std::vector<std::string>& tokens);

        void generateAutoencoderPairs();

        void generateMergePredictorPairs();

        std::vector<float> createOneHotVector(const std::string& token) const;

        bool shouldSubsample(const std::string& token) const;

        std::vector<float> prepareMergeInput(const std::string& token1, const std::string& token2) const;

        void extractEmbeddings();

        void generateTrainingBatch(std::shared_ptr<Matrix>& input, std::shared_ptr<Matrix>& output, int batch_size);

        void generateTestBatch(std::shared_ptr<Matrix>& input, std::shared_ptr<Matrix>& output, int batch_size);

        void normalizeVector(std::vector<float>& vec) const;

        std::string getStrategyName(EmbeddingStrategy strategy) const;
    };

};
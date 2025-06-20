#pragma once

#include <vector>
#include <string>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace NNGL {

    enum VocabReductionStrategy {
        REMOVE_LEAST_FREQUENT,
        REMOVE_MOST_RECENT,
        REMOVE_LONGEST,
        REMOVE_SHORTEST
    };

    // Hash functors
    struct PairHash {
        template <typename T1, typename T2>
        std::size_t operator()(const std::pair<T1, T2>& p) const noexcept;
    };

    struct VectorStringHash {
        size_t operator()(const std::vector<std::string>& vec) const noexcept;
    };

    struct VectorStringEqual {
        bool operator()(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs) const noexcept;
    };

    class BytePairEncodingVectorMapper {
    public:
        using TokenPair = std::pair<std::string, std::string>;

        // Constructor
        explicit BytePairEncodingVectorMapper(int vector_dim = 100,
            int save_interval = 1000) noexcept
            : vector_dim_(vector_dim), save_interval_(save_interval),
            rng_(std::random_device{}()), current_merge_step_(0), is_trained_(false) {}

        // Training methods
        void train(const std::vector<std::string>& corpus, const std::string& checkpoint_prefix = "checkpoint", int target_merges = 0);
        void train(const std::string& filename, const std::string& checkpoint_prefix = "checkpoint", int target_merges = 0);
        // Reset training state
        void resetTraining();

        // BPE methods
        std::vector<std::string> encode(const std::string& word) const noexcept;
        std::set<std::string> getAllTokens() const;

        // Vector methods
        void addToken(const std::string& token);
        void addToken(const std::string& token, const std::vector<float>& vector);
        std::vector<float> getVector(const std::string& token) const;
        std::vector<float>& getVectorRef(const std::string& token);
        bool hasToken(const std::string& token) const noexcept;
        std::vector<std::string> getTokens() const;

        // Similarity and processing
        float getSimilarity(const std::string& token1, const std::string& token2) const;
        std::vector<std::pair<std::string, float>> findSimilar(const std::string& token, int top_k) const;
        std::vector<std::vector<float>> encodeToVectors(const std::string& word) const;
        bool updateVector(const std::string& token, const std::vector<float>& new_vector);

        // Save/Load complete model
        bool save(const std::string& bpe_filename, const std::string& embeddings_filename) const;
        bool load(const std::string& bpe_filename, const std::string& embeddings_filename);

        // Getters
        int getVectorDim() const noexcept { return vector_dim_; }
        int getCurrentMerges() const noexcept { return static_cast<int>(merges_.size()); }
        bool isTrained() const noexcept { return is_trained_; }
        bool reduceVocabulary(int max_tokens, VocabReductionStrategy strategy);

    private:

        // BPE data
        std::unordered_map<std::vector<std::string>, int, VectorStringHash, VectorStringEqual> vocab_;
        std::unordered_set<std::string> base_tokens_;
        std::vector<TokenPair> merges_;

        // Vector mapping data
        int vector_dim_;
        std::unordered_map<std::string, std::vector<float>> token_vectors_;
        mutable std::mt19937 rng_;

        // Training state
        int save_interval_;
        int corpus_chunk_size_ = 128;
        int current_merge_step_, min_merge_frequency_;
        bool is_trained_;

        // BPE helper methods
        void clean_word(std::string& word) noexcept;
        void build_vocab(const std::vector<std::string>& corpus);
        void update_vocab(const std::vector<std::string>& new_corpus);
        std::unordered_map<TokenPair, int, PairHash> get_pair_frequencies() const noexcept;
        void merge_pair(const TokenPair& pair_to_merge);

        // Vector helper methods
        std::vector<float> initRandomVector();
        void normalizeVector(std::vector<float>& vec);
        static float dotProduct(const std::vector<float>& a, const std::vector<float>& b);
        static float magnitude(const std::vector<float>& vec);

        // File I/O helpers
        static void write_string(std::ofstream& out, const std::string& str);
        static std::string read_string(std::ifstream& in);

        // Checkpoint methods
        bool saveCheckpoint(const std::string& checkpoint_prefix, int merge_step) const;
        bool loadCheckpoint(const std::string& checkpoint_prefix);
        std::string getCheckpointFilename(const std::string& prefix, const std::string& suffix) const;

        // Training state management
        void initializeVectorsForNewTokens();
        void updateVectorsAfterMerge(const TokenPair& merged_pair);
        void performTrainingSteps(const std::string& checkpoint_prefix, int target_merges);
        void applyMergeToTokens(std::vector<std::string>& tokens, const TokenPair& merge) const;

        void rebuildVocabulary();
        bool removeTokens(const std::vector<std::string>& tokens_to_remove);

        std::vector<std::string> selectTokensByFrequency(int max_tokens, bool keep_frequent) const;
        std::vector<std::string> selectTokensByRecency(int max_tokens) const;
        std::vector<std::string> selectTokensByLength(int max_tokens, bool remove_short) const;

        void setMinMergeFrequency(int min_freq);
        void setSaveInterval(int interval);
        void setCorpusChunkSize(size_t chunk_size);
        int getMinMergeFrequency() const;
        int getSaveInterval() const;
        size_t getCorpusChunkSize() const;
        int getCurrentMergeStep() const;
        size_t getVocabularySize() const;
        size_t getNumMerges() const;
        void printStatistics() const;
    };
}  // namespace NNGL
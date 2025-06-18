#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace NNGL {

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

    class BytePairEncoding {
    public:
        explicit BytePairEncoding(int max_merges = 1000) noexcept;

        void train(const std::vector<std::string>& corpus);
        void train(const std::string& filename);

        std::vector<std::string> encode(const std::string& word) const noexcept;

        bool save(const std::string& filename) const;
        bool load(const std::string& filename);

    private:
        using TokenPair = std::pair<std::string, std::string>;

        int max_merges_;
        std::unordered_map<std::vector<std::string>, int, VectorStringHash, VectorStringEqual> vocab_;
        std::vector<TokenPair> merges_;

        static void clean_word(std::string& word) noexcept;
        void build_vocab(const std::vector<std::string>& corpus);
        std::unordered_map<TokenPair, int, PairHash> get_pair_frequencies() const noexcept;
        void merge_pair(const TokenPair& pair_to_merge);
        static std::vector<std::string> split(const std::string& str);

        static void write_string(std::ofstream& out, const std::string& str);
        static std::string read_string(std::ifstream& in);
    };

}  // namespace NNGL


#include <fstream>
#include <vector>
#include <cstdint>


uint32_t readUint32(std::ifstream& ifs) {
    uint32_t val = 0;
    for (int i = 0; i < 4; ++i)
        val = (val << 8) | ifs.get();
    return val;
}

namespace MLGL {
    namespace MNIST {
        std::vector<std::vector<uint8_t>> loadImages(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open file");

            readUint32(file); // magic number
            uint32_t numImages = readUint32(file);
            uint32_t rows = readUint32(file);
            uint32_t cols = readUint32(file);

            std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(rows * cols));

            for (uint32_t i = 0; i < numImages; ++i)
                file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);

            return images;
        }

        std::vector<uint8_t> loadLabels(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open label file");

            readUint32(file); // magic number
            uint32_t numLabels = readUint32(file);

            std::vector<uint8_t> labels(numLabels);
            file.read(reinterpret_cast<char*>(labels.data()), numLabels);
            return labels;
        }
    }
}

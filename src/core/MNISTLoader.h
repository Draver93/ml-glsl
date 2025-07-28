#pragma once

#include <string>
#include <vector>

namespace MLGL {
	namespace MNIST {
        std::vector<std::vector<uint8_t>> loadImages(const std::string& filename);
		std::vector<uint8_t> loadLabels(const std::string& filename);
	}
}

/*#define GLAD_GLX 0  // Disable GLX support for Windows
extern "C" {
#include <glad/glad.h>
}

#include <GLFW/glfw3.h>
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "GLFW initialization failed!" << std::endl;
        return -1;
    }

    // Create GLFW windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL with GLFW & GLAD", nullptr, nullptr);
    if (!window) {
        std::cerr << "GLFW window creation failed!" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Initialize GLAD to load OpenGL functions
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD!" << std::endl;
        return -1;
    }

    // Set OpenGL viewport size
    glViewport(0, 0, 800, 600);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Process events
        glfwPollEvents();

        // Render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Cleanup and close the window
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}*/
#define NOMINMAX
#include <windows.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <vector>
#include <functional>
#include <algorithm>


void resetCursor() {
#ifdef _WIN32
    // Windows console API (best way)
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = { 0, 0 };
    SetConsoleCursorPosition(hConsole, coord);
#else
    // ANSI escape sequence for UNIX-like systems
    std::cout << "\033[H"; // Move cursor to top-left
#endif
}

using namespace std;

const std::string colors[] = {
    "\033[48;5;17m",   // deep blue
    "\033[48;5;18m",
    "\033[48;5;19m",
    "\033[48;5;20m",
    "\033[48;5;21m",   // blue
    "\033[48;5;38m",   // teal
    "\033[48;5;44m",   // cyan
    "\033[48;5;51m",   // light cyan
    "\033[48;5;87m",   // white-blue
    "\033[48;5;123m",  // white-cyan
    "\033[48;5;159m",  // white
    "\033[48;5;190m",  // light yellow
    "\033[48;5;226m",  // yellow
    "\033[48;5;220m",  // orange
    "\033[48;5;202m",  // orange-red
    "\033[48;5;196m",  // bright red
    "\033[0m"          // reset
};

void printHeatmap(const std::vector<std::vector<float>>& matrix) {
    float min_val = matrix[0][0], max_val = matrix[0][0];

    // Find min and max values for normalization
    for (auto& row : matrix)
        for (auto val : row) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

    for (auto& row : matrix) {
        for (auto val : row) {
            // Normalize value between 0 and 1
            float norm = (val - min_val) / (max_val - min_val + 1e-7f);
            int color_idx = static_cast<int>(norm * (sizeof(colors) / sizeof(colors[0]) - 2));
            std::cout << colors[color_idx] << "  " << colors[6]; // print 2 spaces with bg color, then reset
        }
        std::cout << "\n";
    }

    std::cout << "\033[0m\n";
}



struct ActivationFunction {
    std::function<float(float)> func; // should now expect pre-activation
    std::function<float(float)> dfunc; // should now expect pre-activation
    std::function<float(int, int)> weight_initializer;    // weight initializer: (in_size, out_size)
};

ActivationFunction sigmoid_fn = {
    [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
    [](float z) {
        float y = 1.0f / (1.0f + std::exp(-z));
        return y * (1 - y);
    },
    [](int in_size, int out_size) { float range = std::sqrt(6.0f / (in_size + out_size)); return ((float)rand() / RAND_MAX) * 2 * range - range; } //xavier_init
};

ActivationFunction tanh_fn = {
    [](float x) { return std::tanh(x); },
    [](float z) {
        float y = std::tanh(z);
        return 1.0f - y * y;
    },
    [](int in_size, int out_size) { float range = std::sqrt(6.0f / (in_size + out_size)); return ((float)rand() / RAND_MAX) * 2 * range - range; } //xavier_init
};

ActivationFunction relu_fn = {
    [](float x) { return std::max(0.0f, x); },
    [](float z) { return z > 0 ? 1.0f : 0.0f; },
    [](int in_size, int /*out_size*/) { float stddev = std::sqrt(2.0f / in_size); return ((float)rand() / RAND_MAX) * 2 * stddev - stddev; } //he_init
};

ActivationFunction leaky_relu_fn = {
    [](float x) { return x > 0 ? x : 0.01f * x; },
    [](float z) { return z > 0 ? 1.0f : 0.01f; },
    [](int in_size, int /*out_size*/) { float stddev = std::sqrt(2.0f / in_size); return ((float)rand() / RAND_MAX) * 2 * stddev - stddev; } //he_init
};

ActivationFunction identity_fn = {
    [](float x) { return x; },
    [](float) { return 1.0f; },
    [](int in_size, int out_size) { float range = std::sqrt(6.0f / (in_size + out_size)); return ((float)rand() / RAND_MAX) * 2 * range - range; } //xavier_init
};

struct Layer {
    std::vector<std::vector<float>> weights;  // [in][out]

    ActivationFunction activationFunction;
    std::vector<float> biases;                // [out]
    std::vector<float> activations;           // [out]
    std::vector<float> pre_activations;       // [out]
    std::vector<float> deltas;                // [out]
};


// Softmax activation function
std::vector<float> softmax(const std::vector<float>& x) {
    float max = *std::max_element(x.begin(), x.end());
    float sum = 0.0;
    std::vector<float> result(x.size());

    // Calculate the softmax values
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max);
        sum += result[i];
    }

    // Normalize to get probabilities
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }

    return result;
}

// Random float between -1 and 1
float random_norm_val() {
    return ((float)rand() / RAND_MAX) * 2 - 1; // Random weight between -1 and 1
}

Layer create_layer(int input_size, int output_size, ActivationFunction activation_func) {
    Layer layer;
    layer.weights.resize(input_size, std::vector<float>(output_size));
    layer.biases.resize(output_size);
    layer.activations.resize(output_size);
    layer.activationFunction = activation_func;
    layer.pre_activations.resize(output_size);
    layer.deltas.resize(output_size);

    for (auto& row : layer.weights)
        for (auto& w : row)
            w = activation_func.weight_initializer(input_size, output_size); //random_norm_val();

    for (auto& b : layer.biases)
        b = activation_func.weight_initializer(input_size, output_size); //random_norm_val();

    return layer;
}

// Forward pass
std::vector<float> forward_pass(std::vector<Layer>& layers, const std::vector<float>& input) {
    std::vector<float> current_data = input;
    for (auto& layer : layers) {

        for (size_t j = 0; j < layer.biases.size(); ++j) {

            float weighted_data = layer.biases[j];
            for (size_t i = 0; i < current_data.size(); ++i)
                weighted_data += current_data[i] * layer.weights[i][j];

            layer.pre_activations[j] = weighted_data;
            layer.activations[j] = layer.activationFunction.func(layer.pre_activations[j]);
        }
        current_data = layer.activations;
    }

    return current_data;
}

// Output layer delta (cross-entropy + sigmoid case)
void compute_output_deltas(Layer& output_layer, const std::vector<float>& target) {
    for (size_t i = 0; i < target.size(); ++i) {
        output_layer.deltas[i] = (output_layer.activations[i] - target[i]) * output_layer.activationFunction.dfunc(output_layer.pre_activations[i]);
    }
}

// Hidden layer delta
void compute_hidden_deltas(Layer& current, const Layer& next) {
    for (size_t i = 0; i < current.activations.size(); ++i) {
        float error = 0.0f;
        for (size_t j = 0; j < next.deltas.size(); ++j)
            error += next.weights[i][j] * next.deltas[j];
        current.deltas[i] = error * current.activationFunction.dfunc(current.pre_activations[i]);
    }
}

// Weight + bias update
void update_weights_and_biases(Layer& layer, const std::vector<float>& input, float lr) {
    for (size_t i = 0; i < layer.weights.size(); ++i)
        for (size_t j = 0; j < layer.weights[0].size(); ++j)
            layer.weights[i][j] -= lr * layer.deltas[j] * input[i];

    for (size_t j = 0; j < layer.biases.size(); ++j)
        layer.biases[j] -= lr * layer.deltas[j];
}

float round_to_step(float value, float step) {
    return std::round(value / step) * step;
}

const int STEPS = 10000000;

int main() {
    srand(time(nullptr));

    int steps_left = STEPS;
    float learning_rate = 0.1f;
    std::vector<float> input(2);
    std::vector<float> target(1);
    bool data_state = false;

    auto update_training_data = [&]() {

        //lets set the input data
        float input_val[2] = { 3.14f * random_norm_val(),  3.14f * random_norm_val() };
        float step = 0.5f;
        //input_val[0] = round_to_step(input_val[0], step);
        //input_val[1] = round_to_step(input_val[1], step);


        float target_state = std::sin(input_val[0]) * std::sin(input_val[1]);

        input[0] = input_val[0] / 3.14f;
        input[1] = input_val[1] / 3.14f;

        target = { (float)target_state };
    };

    std::vector<Layer> layers;
    int internal_layer_size = 30;
    layers.push_back(create_layer(input.size(), internal_layer_size, tanh_fn)); // input → hidden1

    layers.push_back(create_layer(internal_layer_size, internal_layer_size, relu_fn));
    layers.push_back(create_layer(internal_layer_size, internal_layer_size, leaky_relu_fn));
    layers.push_back(create_layer(internal_layer_size, internal_layer_size, sigmoid_fn));
    layers.push_back(create_layer(internal_layer_size, internal_layer_size, tanh_fn));

    layers.push_back(create_layer(internal_layer_size, target.size(), identity_fn)); // hidden9 → output

    while (steps_left > 0) {
        update_training_data();

        // FORWARD PASS
        forward_pass(layers, input);

        // BACKPROPAGATION
        compute_output_deltas(layers.back(), target);

        for (int i = (int)layers.size() - 2; i >= 0; --i)
            compute_hidden_deltas(layers[i], layers[i + 1]);

        // WEIGHT & BIAS UPDATES
        for (size_t i = 0; i < layers.size(); ++i) {
            const std::vector<float>& input_data = (i == 0) ? input : layers[i - 1].activations;
            update_weights_and_biases(layers[i], input_data, learning_rate);
        }

        if (steps_left % 100000 == 0) {
            resetCursor();
            //std::cout << "\033[2J\033[1;1H";
            for (auto& it : layers) {
                printHeatmap(it.weights);
                std::cout << std::endl;
            }

            float loss = 0;
            auto output = layers.back().activations;
            for (int i = 0; i < output.size(); ++i)
                loss += (output[i] - target[i]) * (output[i] - target[i]);
            std::cout << "Step: " << steps_left << " LR: "<< learning_rate << " Loss: " << loss << "  " << std::endl;
        }

        //learning rate decay
        if (steps_left % (STEPS / 20) == 0 ) {
            learning_rate *= 0.9f;
        }

        steps_left--;
    }



    while (true) {
        std::cout << "Enter two numbers (a b): ";
        float a = 0, b = 0;
        if (!(std::cin >> a >> b)) {
            std::cout << "Invalid input. Exiting...\n";
            break;
        }

        float user_input[2] = {a, b};

        input[0] = user_input[0] / 3.14f;
        input[1] = user_input[1] / 3.14f;

        forward_pass(layers, input);

        auto result = layers.back().activations;// softmax(layers.back().activations);
        // Output result
        std::cout << "Output: ";
        for (float o : result)
            std::cout << o << " " ;

        std::cout << "Should be: " << std::sin(a) * std::sin(b) << std::endl;
    }

    return 0;
}
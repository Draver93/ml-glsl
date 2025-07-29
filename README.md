# ML-GLSL

ML-GLSL is a C++ project for machine learning and graphics tasks using GLSL (OpenGL Shading Language). It provides tools for tokenizing GLSL/text data and training transformer-based language models, leveraging OpenGL compute for efficient neural network operations.

---

## Features

- **Tokenizer**: Byte Pair Encoding (BPE) tokenizer for text and code, with training, tokenization, info, and vocabulary reduction modes.
- **Transformer**: Transformer language model for sequence prediction, text/code generation, and evaluation, with flexible training and inference options.
- **OpenGL Compute**: Uses OpenGL and GPU acceleration for neural network computations.
- **Extensive CLI**: Both tokenizer and transformer provide detailed command-line interfaces for all operations.

---

## Project Structure

- `src/tokenizer/main.cpp` — BPE tokenizer CLI and logic.
- `src/transformer/main.cpp` — Transformer model CLI and logic.
- `src/old_main.cpp` — Legacy and experimental neural network and transformer code, including MNIST digit recognition and unit tests.
- `external/` — Third-party dependencies (GLM, GLFW, Glad, etc.).
- `build_win.bat`, `build_linux.sh` — Build scripts for Windows and Linux.
- `premake5.lua` — Build configuration for Premake.

---

## External Dependencies

- **GLM**: Math library for graphics ([external/glm](external/glm))
- **GLFW**: Window/context management ([external/glfw](external/glfw))
- **Glad**: OpenGL loader ([external/glad](external/glad))

---

## Build Instructions

### Windows

1. Run `build_win.bat` to build with Visual Studio.
2. Open `ml-glsl.sln` for development.

### Linux

1. Run `build_linux.sh` to build using the shell script.
2. Ensure `cmake` and `gcc` are installed.

### Using Premake

Generate project files with:

```sh
premake5 vs2019
```
Replace `vs2019` with your desired platform.

---

## Usage

### Tokenizer

The tokenizer supports the following modes:

- **Train**: Train a new BPE model or append to an existing one.
- **Tokenize**: Tokenize input text or files using a trained BPE model.
- **Info**: Display information about a BPE model.
- **Reduce**: Reduce the vocabulary size of a BPE model.

#### Example Commands

- Train a new BPE model:
  ```sh
  ml-glsl-tokenizer --mode train --checkpoint model.bpe --input file1.txt,file2.txt
  ```
- Append to an existing model:
  ```sh
  ml-glsl-tokenizer --mode train --checkpoint model.bpe --input new_data.txt --append
  ```
- Tokenize text:
  ```sh
  ml-glsl-tokenizer --mode tokenize --checkpoint model.bpe --text "Hello world!"
  ```
- Tokenize a file:
  ```sh
  ml-glsl-tokenizer --mode tokenize --checkpoint model.bpe --input input.txt --output tokens.txt
  ```
- Show model info:
  ```sh
  ml-glsl-tokenizer --mode info --checkpoint model.bpe
  ```
- Reduce vocabulary size:
  ```sh
  ml-glsl-tokenizer --mode reduce --checkpoint model.bpe --vocab-size 10000
  ```
- Reduce and save to new file:
  ```sh
  ml-glsl-tokenizer --mode reduce --checkpoint model.bpe --vocab-size 10000 --output small_model.bpe
  ```

#### Tokenizer Options

- `--mode <train|tokenize|info|reduce>`
- `--checkpoint <path>`
- `--input <file1,file2,...>` (for train) or `<file>` (for tokenize)
- `--text <text>` (for tokenize)
- `--output <file>`
- `--append` (for train)
- `--vocab-size <size>`
- `--merge-limit <limit>`
- `--no-special-tokens`
- `--no-ascii-tokens`
- `--verbose`
- `--help`

---

### Transformer

The transformer supports the following modes:

- **Train**: Train a new transformer model.
- **Generate**: Generate text/code from a prompt.
- **Evaluate**: Evaluate the model on test data.
- **Interactive**: Interactive prompt/response mode.
- **Info**: Show model information.

#### Example Commands

- Train a new transformer model:
  ```sh
  ml-glsl-transformer --mode train --bpe tokenizer.bpe --model model.gpt --input data.txt
  ```
- Generate text:
  ```sh
  ml-glsl-transformer --mode generate --model model.gpt --prompt "Once upon a time"
  ```
- Evaluate the model:
  ```sh
  ml-glsl-transformer --mode evaluate --model model.gpt --input test_data.txt
  ```
- Interactive mode:
  ```sh
  ml-glsl-transformer --mode interactive --model model.gpt
  ```
- Show model info:
  ```sh
  ml-glsl-transformer --mode info --model model.gpt
  ```

#### Transformer Options

- `--mode <train|generate|info|evaluate|interactive>`
- `--model <path>`
- `--bpe <path>`
- `--input <file1,file2,...>`
- `--output <file>`
- `--prompt <text>`
- `--d-model <size>`
- `--d-hidden <size>`
- `--seq-len <length>`
- `--epochs <num>`
- `--lr <rate>`
- `--lr-decay <factor>`
- `--lr-decay-steps <steps>`
- `--progress-interval <steps>`
- `--eval-interval <steps>`
- `--early-stopping <patience>`
- `--target-loss <loss>`
- `--max-tokens <num>`
- `--temperature <temp>`
- `--top-k <k>`
- `--no-eos`
- `--num-prompts <num>`
- `--show-loss-trend`
- `--verbose`
- `--help`

---

## Testing

The project includes several test cases in the form of `.bmp` files (e.g., `test_case_1.bmp`, `test_case_2.bmp`). These can be used to validate the functionality of the neural network and transformer components. The legacy `old_main.cpp` also contains comprehensive unit tests for matrix operations, layer normalization, attention, and more.

---

## Documentation

### GLM

GLM documentation is available in `external/glm/doc/api`.

### GLFW

GLFW documentation is available in `external/glfw/docs`.

---

## License

This project is proprietary and not licensed for redistribution or commercial use. All rights reserved. Third-party libraries may have their own licenses; see their respective directories for details.

---

## Contact

For questions or issues, please open an issue in the repository or contact the maintainers.
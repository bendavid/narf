#ifndef NARF_TFLITEUTILS_H
#define NARF_TFLITEUTILS_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include <tbb/task_arena.h>
#include <algorithm>

namespace narf {

    class tflite_interpreter_data {
    public:
        tflite_interpreter_data(tflite::InterpreterBuilder &builder, const std::string &signature) {
            if (builder(&interpreter_) != kTfLiteOk) {
                throw std::runtime_error("Failed to build interpreter");
            }

            runner_ = interpreter_->GetSignatureRunner(signature.c_str());
            if (runner_ == nullptr) {
                throw std::runtime_error("Failed to get signature runner");
            }

            if (runner_->AllocateTensors() != kTfLiteOk) {
                throw std::runtime_error("Failed to allocate tensors");
            }

            for (auto const &input_name : runner_->input_names()) {
                TfLiteTensor *input = runner_->input_tensor(input_name);
                if (input == nullptr) {
                    throw std::runtime_error("Failed to find input tensor");
                }
                inputs_.push_back(input);
            }

            for (auto const &output_name : runner_->output_names()) {
                const TfLiteTensor *output = runner_->output_tensor(output_name);
                if (output == nullptr) {
                    throw std::runtime_error("Failed to find output tensor");
                }
                outputs_.push_back(output);
            }

        }

        tflite::SignatureRunner *runner() { return runner_; }
        const std::vector<TfLiteTensor*> inputs() { return inputs_; }
        const std::vector<const TfLiteTensor*> outputs() { return outputs_; }

        template <typename inputs_t, typename outputs_t>
        void operator() (const inputs_t &inputs, outputs_t &outputs) {

            auto const &tfinputs = inputs_;
            auto const &tfoutputs = outputs_;

            auto fill_inputs = [&tfinputs](auto const&... tensors) {
                auto fill_input = [](TfLiteTensor *tfinput, auto const &tensor) {
                    using scalar_t = typename std::decay_t<decltype(tensor)>::Scalar;

                    const scalar_t *src = tensor.data();
                    scalar_t *dest = tflite::GetTensorData<scalar_t>(tfinput);

                    std::copy(src, src + tensor.size(), dest);
                };

                std::size_t i = 0;
                (fill_input(tfinputs[i++], tensors), ...);
            };

            std::apply(fill_inputs, inputs);

            if (runner_->Invoke() != kTfLiteOk) {
                throw std::runtime_error("Failed to invoke interpreter");
            }

            auto fill_outputs = [&tfoutputs](auto&... tensors) {
                auto fill_output = [](const TfLiteTensor *tfoutput, auto &tensor) {
                    using scalar_t = typename std::decay_t<decltype(tensor)>::Scalar;

                    const scalar_t *src = tflite::GetTensorData<scalar_t>(tfoutput);
                    scalar_t *dest = tensor.data();

                    std::copy(src, src + tensor.size(), dest);
                };

                std::size_t i = 0;
                (fill_output(tfoutputs[i++], tensors), ...);
            };

            std::apply(fill_outputs, outputs);
        }


    private:

        std::unique_ptr<tflite::Interpreter> interpreter_;
        tflite::SignatureRunner* runner_ = nullptr;
        std::vector<TfLiteTensor*> inputs_;
        std::vector<const TfLiteTensor*> outputs_;
    };

    class tflite_helper {
    public:
        tflite_helper(const std::string &filename, const std::string &signature = "serving_default", const unsigned int nslots = 1) :
            model_(tflite::FlatBufferModel::BuildFromFile(filename.c_str())) {

                const unsigned int nslots_actual = std::max(nslots, 1U);

                if (model_ == nullptr) {
                    throw std::runtime_error("Failed to load model");
                }

                tflite::ops::builtin::BuiltinOpResolver resolver;
                tflite::InterpreterBuilder builder(*model_, resolver);
                builder.SetNumThreads(1);

                interpreters_.reserve(nslots_actual);
                for (unsigned int islot = 0; islot < nslots_actual; ++islot) {
                    interpreters_.emplace_back(builder, signature);
                }
        }

        template <typename inputs_t, typename outputs_t>
        void operator() (const inputs_t &inputs, outputs_t &outputs) {
            auto const tbb_slot = std::max(tbb::v1::this_task_arena::current_thread_index(), 0);

            if (tbb_slot >= interpreters_.size()) {
                throw std::runtime_error("Not enough interpreters allocated for number of tbb threads");
            }

            auto &interpreter_data = interpreters_[tbb_slot];

            interpreter_data(inputs, outputs);

        }

    private:
        std::unique_ptr<tflite::FlatBufferModel> model_;
        std::vector<tflite_interpreter_data> interpreters_;
    };

}

#endif

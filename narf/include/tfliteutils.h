#ifndef NARF_TFLITEUTILS_H
#define NARF_TFLITEUTILS_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <algorithm>

namespace narf {

    class tflite_helper {
    public:
        tflite_helper(const std::string &filename, const std::string &signature = "serving_default", const unsigned int nslots = 1) :
            model_(tflite::FlatBufferModel::BuildFromFile(filename.c_str())) {

                unsigned int nslots_actual = std::max(nslots, 1U);

                if (model_ == nullptr) {
                    throw std::runtime_error("Failed to load model");
                }

                tflite::ops::builtin::BuiltinOpResolver resolver;
                tflite::InterpreterBuilder builder(*model_, resolver);
                builder.SetNumThreads(1);

                for (unsigned int islot = 0; islot < nslots_actual; ++islot) {
                    auto &interpreter = interpreters_.emplace_back();
                    if (builder(&interpreter) != kTfLiteOk) {
                        throw std::runtime_error("Failed to build interpreter");
                    }

                    auto &runner = runners_.emplace_back(interpreter->GetSignatureRunner(signature.c_str()));
                    if (runner == nullptr) {
                        throw std::runtime_error("Failed to get signature runner");
                    }

                    if (runner->AllocateTensors() != kTfLiteOk) {
                        throw std::runtime_error("Failed to allocate tensors");
                    }

                    if (islot == 0) {
                        for (auto const &item : interpreter->signature_inputs(signature.c_str())) {
                            input_idxs_.push_back(item.second);
                        }

                        for (auto const &item : interpreter->signature_outputs(signature.c_str())) {
                            output_idxs_.push_back(item.second);
                        }
                    }
                }
        }

        template <typename inputs_t, typename outputs_t>
        void operator() (unsigned int slot, const inputs_t &inputs, outputs_t &outputs) {
            auto &interpreter = interpreters_[slot];
            auto &runner = runners_[slot];


            auto fill_inputs = [this, &interpreter](auto const&... tensors) {
                auto fill_input = [&interpreter](const std::size_t idx, auto const &tensor) {
                    using scalar_t = typename std::decay_t<decltype(tensor)>::Scalar;

                    const scalar_t *src = tensor.data();
                    scalar_t *dest = interpreter->typed_tensor<scalar_t>(idx);

                    std::copy(src, src + tensor.size(), dest);
                };

                std::size_t i = 0;
                (fill_input(input_idxs_[i++], tensors), ...);
            };

            std::apply(fill_inputs, inputs);

            if (runner->Invoke() != kTfLiteOk) {
                throw std::runtime_error("Failed to invoke interpreter");
            }

            auto fill_outputs = [this, &interpreter](auto&... tensors) {
                auto fill_output = [&interpreter](const std::size_t idx, auto &tensor) {
                    using scalar_t = typename std::decay_t<decltype(tensor)>::Scalar;

                    const scalar_t *src = interpreter->typed_tensor<scalar_t>(idx);
                    scalar_t *dest = tensor.data();

                    std::copy(src, src + tensor.size(), dest);
                };

                std::size_t i = 0;
                (fill_output(output_idxs_[i++], tensors), ...);
            };

            std::apply(fill_outputs, outputs);


        }

    private:
        std::unique_ptr<tflite::FlatBufferModel> model_;
        std::vector<std::unique_ptr<tflite::Interpreter>> interpreters_;
        std::vector<tflite::SignatureRunner*> runners_;
        std::vector<int> input_idxs_;
        std::vector<int> output_idxs_;
    };

}

#endif

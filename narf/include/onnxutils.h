#ifndef NARF_ONNXUTILS_H
#define NARF_ONNXUTILS_H

#include "onnxruntime/onnxruntime_cxx_api.h"
#include "traits.h"

namespace narf {


  class onnx_helper {
  public:

    onnx_helper(const std::string &model_path, unsigned int nslots = 1) : meminfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetInterOpNumThreads(1);

      for (unsigned int islot = 0; islot < nslots; ++islot) {
        auto &session = sessions_.emplace_back(env_, model_path.c_str(), options);

        // Ort::Allocator allocator(session, meminfo_);
        Ort::Allocator &allocator = allocators_.emplace_back(session, meminfo_);

        auto const nin = session.GetInputCount();
        auto const nout = session.GetOutputCount();

        if (islot == 0) {

          for (std::size_t iin = 0; iin < nin; ++iin) {
            auto const &namestring = inputNameStrings_.emplace_back(session.GetInputNameAllocated(iin, allocator));
            inputNames_.emplace_back(namestring.get());

            inputShapes_.emplace_back(session.GetInputTypeInfo(iin).GetTensorTypeAndShapeInfo().GetShape());
          }

          for (std::size_t iout = 0; iout < nout; ++iout) {
            auto const &namestring = outputNameStrings_.emplace_back(session.GetOutputNameAllocated(iout, allocator));
            outputNames_.emplace_back(namestring.get());

            outputShapes_.emplace_back(session.GetOutputTypeInfo(iout).GetTensorTypeAndShapeInfo().GetShape());
          }

        }

        auto &inputValues = inputValues_.emplace_back();
        auto &outputValues = outputValues_.emplace_back();

        for (std::size_t iin = 0; iin < nin; ++iin) {
          inputValues.emplace_back(Ort::Value::CreateTensor(allocator,
                                                     inputShapes_[iin].data(),
                                                     inputShapes_[iin].size(),
                                                     session.GetInputTypeInfo(iin).GetTensorTypeAndShapeInfo().GetElementType()));

        }

        for (std::size_t iout = 0; iout < nout; ++iout) {
          outputValues.emplace_back(Ort::Value::CreateTensor(allocator,
                                                     outputShapes_[iout].data(),
                                                     outputShapes_[iout].size(),
                                                     session.GetOutputTypeInfo(iout).GetTensorTypeAndShapeInfo().GetElementType()));

        }
      }
    }

    template<typename input_t, typename output_t>
    void operator() (unsigned int slot, input_t &inputs, output_t &outputs) {

      Ort::Session &session = sessions_[slot];
      auto &inputValues = inputValues_[slot];
      auto &outputValues = outputValues_[slot];

      using Eigen::TensorMap;
      using Eigen::Tensor;

      auto fill_inputs = [&inputValues](auto const&... tensors) {
        auto fill_input = [](Ort::Value &value, auto const &tensor) {
          using scalar_t = typename std::decay_t<decltype(tensor)>::Scalar;

          const scalar_t *src = tensor.data();
          scalar_t *dest = value.GetTensorMutableData<scalar_t>();

          std::copy(src, src + tensor.size(), dest);
        };

        std::size_t i = 0;
        (fill_input(inputValues[i++], tensors), ...);
      };

      std::apply(fill_inputs, inputs);

      session.Run(Ort::RunOptions{nullptr},
              inputNames_.data(),
              inputValues.data(),
              inputNames_.size(),
              outputNames_.data(),
              outputValues.data(),
              outputNames_.size());

      auto fill_outputs = [&outputValues](auto&... tensors) {
        auto fill_output = [](Ort::Value &value, auto &tensor) {
          using scalar_t = typename std::decay_t<decltype(tensor)>::Scalar;

          const scalar_t *src = value.GetTensorMutableData<scalar_t>();
          scalar_t *dest = tensor.data();

          std::copy(src, src + tensor.size(), dest);
        };

        std::size_t i = 0;
        (fill_output(outputValues[i++], tensors), ...);
      };

      std::apply(fill_outputs, outputs);


    }

    template<typename input_t, typename output_t>
    void operator() (input_t &inputs, output_t &outputs) {
      return operator()(0, inputs, outputs);
    }

  private:
    Ort::Env env_;
    Ort::MemoryInfo meminfo_;
    std::vector<Ort::Session> sessions_;
    std::vector<Ort::Allocator> allocators_;
    std::vector<Ort::AllocatedStringPtr> inputNameStrings_;
    std::vector<Ort::AllocatedStringPtr> outputNameStrings_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::vector<std::vector<int64_t>> inputShapes_;
    std::vector<std::vector<int64_t>> outputShapes_;
    std::vector<std::vector<Ort::Value>> inputValues_;
    std::vector<std::vector<Ort::Value>> outputValues_;
  };

  class onnx_helper_alloc {
  public:

    onnx_helper_alloc(const std::string &model_path, unsigned int nslots = 1) : meminfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetInterOpNumThreads(1);

      for (unsigned int islot = 0; islot < nslots; ++islot) {
        sessions_.emplace_back(env_, model_path.c_str(), options);
      }

      Ort::Session &session0 = sessions_.front();
      Ort::Allocator allocator(session0, meminfo_);

      auto const nin = session0.GetInputCount();
      auto const nout = session0.GetOutputCount();

      for (std::size_t i = 0; i < nin; ++i) {
        inputNames_.emplace_back(session0.GetInputNameAllocated(i, allocator).release());
      }

      for (std::size_t i = 0; i < nout; ++i) {
        outputNames_.emplace_back(session0.GetOutputNameAllocated(i, allocator).release());
      }
    }

    template<typename input_t, typename output_t>
    void operator() (unsigned int slot, input_t &inputs, output_t &outputs) {

      auto get_tensors_with_shapes = [](auto&... tensors) {
        auto get_shape = [](const auto &tensor) {
          return narf::tensor_traits<std::decay_t<decltype(tensor)>>::get_sizes();
        };

        return std::make_tuple(std::make_pair(std::ref(tensors), get_shape(tensors))...);
      };

      auto get_values = [this](auto&... tensors_with_shapes) {
        std::array<Ort::Value, sizeof...(tensors_with_shapes)> values = { Ort::Value::CreateTensor<typename std::decay_t<decltype(tensors_with_shapes.first)>::Scalar>(meminfo_, tensors_with_shapes.first.data(), tensors_with_shapes.first.size(), tensors_with_shapes.second.data(), tensors_with_shapes.second.size())... };

        return values;
      };

      auto inputs_with_shapes = std::apply(get_tensors_with_shapes, inputs);
      auto outputs_with_shapes = std::apply(get_tensors_with_shapes, outputs);

      auto inputValues = std::apply(get_values, inputs_with_shapes);
      auto outputValues = std::apply(get_values, outputs_with_shapes);

      Ort::Session &session = sessions_[slot];

      session.Run(Ort::RunOptions{nullptr},
              inputNames_.data(),
              inputValues.data(),
              inputNames_.size(),
              outputNames_.data(),
              outputValues.data(),
              outputNames_.size());
    }

    template<typename input_t, typename output_t>
    void operator() (input_t &inputs, output_t &outputs) {
      return operator()(0, inputs, outputs);
    }

  private:
    Ort::Env env_;
    Ort::MemoryInfo meminfo_;
    std::vector<Ort::Session> sessions_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
  };

}

#endif

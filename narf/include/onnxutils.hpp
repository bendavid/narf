#ifndef NARF_ONNXUTILS_H
#define NARF_ONNXUTILS_H

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tbb/task_arena.h>

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include "onnxruntime/onnxruntime_cxx_api.h"

namespace narf {


  // ONNX Runtime wrapper with persistent per-slot input/output buffers
  // pre-bound through Ort::IoBinding. Each ``operator()`` call copies
  // the caller's Eigen tensors into the persistent ORT-owned buffers
  // via ``Eigen::TensorMap<…, RowMajor>``, runs the session against
  // the binding, and copies the outputs back out the same way.
  //
  // The TensorMap-based copy enforces ORT's row-major contract
  // regardless of the layout the caller passed in: assigning a
  // (possibly ColMajor) Eigen tensor into a RowMajor TensorMap
  // transparently reindexes element-by-element, so passing a ColMajor
  // multi-axis tensor produces correct numerical results instead of a
  // silent layout scramble.
  //
  // Construction has two flavours:
  //   * static-shape:  shapes are queried from the session metadata.
  //                    Throws if any input or output shape contains a
  //                    dynamic axis (-1).
  //   * explicit-shape: shapes are supplied by the caller and used to
  //                    allocate the persistent buffers. Use this when
  //                    the model has dynamic axes that the caller is
  //                    pinning to concrete sizes at construction.
  class onnx_helper {
  public:

    // Static-shape constructor. Reads each input/output's shape from
    // ``session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo()`` and
    // throws if any contains a dynamic axis (-1).
    onnx_helper(const std::string &model_path, unsigned int nslots = 1) {
      init_(model_path, /*input_shapes_explicit=*/nullptr,
            /*output_shapes_explicit=*/nullptr, nslots);
    }

    // Explicit-shape constructor. ``input_shapes`` and ``output_shapes``
    // must each be ``session.GetInputCount()`` / ``GetOutputCount()``
    // long, with concrete (non-negative) dim values. Use this when
    // the model declares one or more dynamic axes; element types are
    // still discovered from the session.
    onnx_helper(const std::string &model_path,
                const std::vector<std::vector<int64_t>> &input_shapes,
                const std::vector<std::vector<int64_t>> &output_shapes,
                unsigned int nslots = 1) {
      init_(model_path, &input_shapes, &output_shapes, nslots);
    }

    template<typename input_t, typename output_t>
    void operator() (const input_t &inputs, output_t &outputs) {
      // Slot indexed by the TBB task arena thread index rather than
      // RDataFrame's per-graph slot. RDataFrame slots are not unique
      // across multiple RDataFrame loops scheduled by ``RunGraphs`` in
      // the same task arena, whereas the TBB thread index is.
      auto const tbb_slot = std::max(tbb::v1::this_task_arena::current_thread_index(), 0);

      if (static_cast<std::size_t>(tbb_slot) >= sessions_.size()) {
        throw std::runtime_error("Not enough sessions allocated for number of tbb threads");
      }

      Ort::Session &session = sessions_[tbb_slot];
      auto &inputValues = inputValues_[tbb_slot];
      auto &outputValues = outputValues_[tbb_slot];
      auto &binding = bindings_[tbb_slot];

      auto fill_inputs = [this, &inputValues](auto const&... tensors) {
        std::size_t i = 0;
        auto one = [&](auto const &t) { fill_into_ort_(inputValues[i++], t); };
        (one(tensors), ...);
      };
      std::apply(fill_inputs, inputs);

      session.Run(Ort::RunOptions{nullptr}, binding);

      auto fill_outputs = [this, &outputValues](auto&... tensors) {
        std::size_t i = 0;
        auto one = [&](auto &t) { fill_from_ort_(outputValues[i++], t); };
        (one(tensors), ...);
      };
      std::apply(fill_outputs, outputs);
    }

  private:
    // ``env_`` must outlive every session built from it (sessions
    // don't hold their own ref). The default-options allocator is an
    // ``Unowned<OrtAllocator>`` wrapper around an ORT-library-owned
    // singleton — destroying the wrapper does not free the
    // underlying allocator, so we can construct it as a local in
    // ``init_`` and let it go; the ``Ort::Value::~Value`` calls of
    // our member vectors will free their buffers against the same
    // singleton at class-destruction time.
    Ort::Env env_;
    std::vector<Ort::Session> sessions_;
    std::vector<std::vector<Ort::Value>> inputValues_;
    std::vector<std::vector<Ort::Value>> outputValues_;
    std::vector<Ort::IoBinding> bindings_;

    void init_(const std::string &model_path,
               const std::vector<std::vector<int64_t>> *input_shapes_explicit,
               const std::vector<std::vector<int64_t>> *output_shapes_explicit,
               unsigned int nslots) {
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetInterOpNumThreads(1);

      // Global ORT-owned default CPU allocator. ``Unowned`` wrapper,
      // so this local can die at end of ``init_`` — the OrtValues
      // we hand to ``CreateTensor`` below hold a pointer to the
      // underlying singleton, not to this wrapper.
      Ort::AllocatorWithDefaultOptions allocator;

      // Locals: filled on the first slot iteration and reused for
      // the rest. AllocatedStringPtrs free themselves through the
      // allocator at end of scope; the underlying allocator is the
      // ORT-library-owned singleton, alive for the program's lifetime.
      std::vector<Ort::AllocatedStringPtr> inputNameStrings;
      std::vector<Ort::AllocatedStringPtr> outputNameStrings;
      std::vector<const char*> inputNames;
      std::vector<const char*> outputNames;
      std::vector<std::vector<int64_t>> inputShapes;
      std::vector<std::vector<int64_t>> outputShapes;
      std::vector<ONNXTensorElementDataType> inputElementTypes;
      std::vector<ONNXTensorElementDataType> outputElementTypes;

      for (unsigned int islot = 0; islot < nslots; ++islot) {
        auto &session = sessions_.emplace_back(env_, model_path.c_str(), options);

        auto const nin = session.GetInputCount();
        auto const nout = session.GetOutputCount();

        if (islot == 0) {
          if (input_shapes_explicit && input_shapes_explicit->size() != nin) {
            throw std::invalid_argument(
                "onnx_helper: input_shapes size does not match model input count");
          }
          if (output_shapes_explicit && output_shapes_explicit->size() != nout) {
            throw std::invalid_argument(
                "onnx_helper: output_shapes size does not match model output count");
          }

          // ``TypeInfo`` owns the underlying ``OrtTypeInfo``; the
          // ``ConstTensorTypeAndShapeInfo`` we get from it is a
          // non-owning view that must not outlive the parent.
          for (std::size_t iin = 0; iin < nin; ++iin) {
            auto const &namestring = inputNameStrings.emplace_back(
                session.GetInputNameAllocated(iin, allocator));
            inputNames.emplace_back(namestring.get());

            auto type_info = session.GetInputTypeInfo(iin);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            inputElementTypes.emplace_back(info.GetElementType());

            std::vector<int64_t> shape = input_shapes_explicit
                ? (*input_shapes_explicit)[iin]
                : info.GetShape();
            for (auto d : shape) {
              if (d < 0) {
                throw std::runtime_error(
                    std::string("onnx_helper: input '") + inputNames.back() +
                    "' has a dynamic axis; use the explicit-shape constructor");
              }
            }
            inputShapes.emplace_back(std::move(shape));
          }

          for (std::size_t iout = 0; iout < nout; ++iout) {
            auto const &namestring = outputNameStrings.emplace_back(
                session.GetOutputNameAllocated(iout, allocator));
            outputNames.emplace_back(namestring.get());

            auto type_info = session.GetOutputTypeInfo(iout);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            outputElementTypes.emplace_back(info.GetElementType());

            std::vector<int64_t> shape = output_shapes_explicit
                ? (*output_shapes_explicit)[iout]
                : info.GetShape();
            for (auto d : shape) {
              if (d < 0) {
                throw std::runtime_error(
                    std::string("onnx_helper: output '") + outputNames.back() +
                    "' has a dynamic axis; use the explicit-shape constructor");
              }
            }
            outputShapes.emplace_back(std::move(shape));
          }
        }

        auto &inputValues = inputValues_.emplace_back();
        auto &outputValues = outputValues_.emplace_back();

        for (std::size_t iin = 0; iin < nin; ++iin) {
          inputValues.emplace_back(Ort::Value::CreateTensor(
              allocator,
              inputShapes[iin].data(),
              inputShapes[iin].size(),
              inputElementTypes[iin]));
        }
        for (std::size_t iout = 0; iout < nout; ++iout) {
          outputValues.emplace_back(Ort::Value::CreateTensor(
              allocator,
              outputShapes[iout].data(),
              outputShapes[iout].size(),
              outputElementTypes[iout]));
        }

        // Pre-bind every input/output Ort::Value to this slot's
        // session-local IoBinding. Subsequent ``session.Run`` calls
        // re-use the same buffers — no per-call Ort::Value
        // construction, no per-call name array setup, no copies on
        // the ORT side.
        auto &binding = bindings_.emplace_back(session);
        for (std::size_t iin = 0; iin < nin; ++iin) {
          binding.BindInput(inputNames[iin], inputValues[iin]);
        }
        for (std::size_t iout = 0; iout < nout; ++iout) {
          binding.BindOutput(outputNames[iout], outputValues[iout]);
        }
      }
    }

    // Caller-Eigen-tensor -> ORT buffer copy via a RowMajor TensorMap
    // assignment. The shape comes directly from the caller's tensor;
    // the ORT-side buffer was allocated to ``inputShapes_[i]`` at
    // construction and the caller is responsible for matching it.
    // Works for any caller layout (ColMajor or RowMajor): Eigen's
    // cross-layout assignment iterates in index space and respects
    // both source and destination strides, so the result is shape-
    // correct regardless of source layout.
    template <typename CallerTensor>
    void fill_into_ort_(Ort::Value &value, const CallerTensor &caller) const {
      using D = std::decay_t<CallerTensor>;
      using scalar_t = typename D::Scalar;
      constexpr int rank = D::NumDimensions;
      Eigen::TensorMap<Eigen::Tensor<scalar_t, rank, Eigen::RowMajor>> ort_view(
          value.template GetTensorMutableData<scalar_t>(), caller.dimensions());
      ort_view = caller;
    }

    // ORT buffer -> caller-Eigen-tensor copy via a RowMajor TensorMap
    // assignment. Same layout-agnostic behaviour as ``fill_into_ort_``.
    template <typename CallerTensor>
    void fill_from_ort_(const Ort::Value &value, CallerTensor &caller) const {
      using D = std::decay_t<CallerTensor>;
      using scalar_t = typename D::Scalar;
      constexpr int rank = D::NumDimensions;
      Eigen::TensorMap<const Eigen::Tensor<scalar_t, rank, Eigen::RowMajor>> ort_view(
          value.template GetTensorData<scalar_t>(), caller.dimensions());
      caller = ort_view;
    }
  };

}

#endif

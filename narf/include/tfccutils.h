#ifndef NARF_TFCCUTILS_H
#define NARF_TFCCUTILS_H

#include <string>
#include <memory>
#include <vector>
#include <tbb/task_arena.h>


namespace tensorflow {
  class SavedModelBundleLite;
  class Tensor;
  class Session;
  class MetaGraphDef;
}

namespace narf {
    class tf_model_data {
    public:
        // tf_model_data(const std::string &model_dir, const std::string &sig_name = "serving_default");
        tf_model_data(const tensorflow::MetaGraphDef &metagraph, const std::string &model_dir, const std::string &sig_name);
        tf_model_data(tf_model_data&&);

        ~tf_model_data();

    private:
        // std::unique_ptr<tensorflow::SavedModelBundleLite> model_;
        std::vector<tensorflow::Tensor> inputs_;
        std::vector<tensorflow::Tensor> outputs_;
        // tensorflow::Session *session_ = nullptr;
        std::unique_ptr<tensorflow::Session> session_;
        int64_t callable_ = 0;
        std::vector<void*> inputdata_;
        std::vector<const void*> outputdata_;
    };




    class tf_helper_pll {
    public:
        tf_helper_pll(const std::string &model_dire, const std::string &sig_name = "serving_default");

        tf_helper_pll(tf_helper_pll&&);


        ~tf_helper_pll();

        void InvokeModel();

        template <typename inputs_t, typename outputs_t>
        void operator() (const inputs_t &inputs, outputs_t &outputs) {

            auto const &tfinputs = inputdata_;
            auto const &tfoutputs = outputdata_;

            auto fill_inputs = [&tfinputs](auto const&... tensors) {
                auto fill_input = [](void *tfinput, auto const &tensor) {
                    using scalar_t = typename std::decay_t<decltype(tensor)>::Scalar;

                    const scalar_t *src = tensor.data();
                    scalar_t *dest = reinterpret_cast<scalar_t*>(tfinput);

                    std::copy(src, src + tensor.size(), dest);
                };

                std::size_t i = 0;
                (fill_input(tfinputs[i++], tensors), ...);
            };

            std::apply(fill_inputs, inputs);

            InvokeModel();

            auto fill_outputs = [&tfoutputs](auto&... tensors) {
                auto fill_output = [](const void *tfoutput, auto &tensor) {
                    using scalar_t = typename std::decay_t<decltype(tensor)>::Scalar;

                    const scalar_t *src = reinterpret_cast<const scalar_t*>(tfoutput);
                    scalar_t *dest = tensor.data();

                    std::copy(src, src + tensor.size(), dest);
                };

                std::size_t i = 0;
                (fill_output(tfoutputs[i++], tensors), ...);
            };

            std::apply(fill_outputs, outputs);
        }

    private:
        std::unique_ptr<tensorflow::SavedModelBundleLite> model_;
        std::vector<tensorflow::Tensor> inputs_;
        std::vector<tensorflow::Tensor> outputs_;
        tensorflow::Session *session_ = nullptr;
        int64_t callable_ = 0;
        std::vector<void*> inputdata_;
        std::vector<const void*> outputdata_;
    };

    class tf_helper {
    public:
        tf_helper(const std::string &model_dir, const std::string &sig_name = "serving_default", const unsigned int nslots = 1);
        // ~tf_helper();

        template <typename inputs_t, typename outputs_t>
        void operator() (const inputs_t &inputs, outputs_t &outputs) {
            auto const tbb_slot = std::max(tbb::v1::this_task_arena::current_thread_index(), 0);

            if (tbb_slot >= model_data_.size()) {
                throw std::runtime_error("Not enough interpreters allocated for number of tbb threads");
            }

            auto &model_data = model_data_[tbb_slot];

            model_data(inputs, outputs);

        }

    private:
        // std::unique_ptr<tensorflow::SavedModelBundleLite> model_;
        std::vector<tf_helper_pll> model_data_;
        // int64_t callable_ = 0;
    };

}

#endif

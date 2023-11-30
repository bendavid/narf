#include "tfccutils.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

// narf::tf_model_data::tf_model_data(const std::string &model_dir, const std::string &sig_name) :
//     model_(std::make_unique<tensorflow::SavedModelBundleLite>()) {
//
//     tensorflow::SessionOptions session_options;
//     tensorflow::RunOptions run_options;
//
//     session_options.config.set_inter_op_parallelism_threads(-1);
//     session_options.config.set_intra_op_parallelism_threads(1);
//
//     auto const status = tensorflow::LoadSavedModel(session_options,
//                                         run_options,
//                                         model_dir,
//                                         {tensorflow::kSavedModelTagServe},
//                                         model_.get());
//
//     if (!status.ok()) {
//        throw std::runtime_error("failed to load saved model from directory");
//     }
//
//     session_ = model_->GetSession();
//
//     auto const &signature = model_->GetSignatures().at(sig_name);
//
//     tensorflow::CallableOptions callopts;
//
//     inputs_.reserve(signature.inputs().size());
//     inputdata_.reserve(signature.inputs().size());
//     for (auto const &input_pair : signature.inputs()) {
//         auto const &input = input_pair.second;
//         callopts.add_feed(input.name());
//         auto &intensor = inputs_.emplace_back(input.dtype(), input.tensor_shape());
//         inputdata_.emplace_back(intensor.data());
//     }
//
//     outputs_.reserve(signature.outputs().size());
//     outputdata_.reserve(signature.outputs().size());
//     for (auto const &output_pair : signature.outputs()) {
//         auto const &output = output_pair.second;
//         callopts.add_fetch(output.name());
//         auto const &outtensor = outputs_.emplace_back(output.dtype(), output.tensor_shape());
//         outputdata_.emplace_back(outtensor.data());
//     }
//
//     auto const statusc = model_->GetSession()->MakeCallable(callopts, &callable_);
//
//     if (!statusc.ok()) {
//         throw std::runtime_error("failed to create callable handle from session");
//     }
//
// }

narf::tf_model_data::tf_model_data(const tensorflow::MetaGraphDef &metagraph, const std::string &model_dir, const std::string &sig_name) {

    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

    session_options.config.set_inter_op_parallelism_threads(-1);
    session_options.config.set_intra_op_parallelism_threads(1);

    // auto const status = tensorflow::LoadSavedModel(session_options,
    //                                     run_options,
    //                                     model_dir,
    //                                     {tensorflow::kSavedModelTagServe},
    //                                     model_.get());
    //
    // if (!status.ok()) {
    //    throw std::runtime_error("failed to load saved model from directory");
    // }
    //
    // session_ = model_->GetSession();

    auto const status = LoadMetagraphIntoSession(session_options, metagraph, &session_);

    if (!status.ok()) {
        throw std::runtime_error("failed to load metagraph into session");
    }

    auto const statusr = RestoreSession(run_options, metagraph, model_dir, &session_);

    if (!statusr.ok()) {
        throw std::runtime_error("failed to restore session state");
    }

    // auto const &signature = model_->GetSignatures().at(sig_name);
    auto const &signature = metagraph.signature_def().at(sig_name);

    tensorflow::CallableOptions callopts;

    inputs_.reserve(signature.inputs().size());
    inputdata_.reserve(signature.inputs().size());
    for (auto const &input_pair : signature.inputs()) {
        auto const &input = input_pair.second;
        callopts.add_feed(input.name());
        auto &intensor = inputs_.emplace_back(input.dtype(), input.tensor_shape());
        inputdata_.emplace_back(intensor.data());
    }

    outputs_.reserve(signature.outputs().size());
    outputdata_.reserve(signature.outputs().size());
    for (auto const &output_pair : signature.outputs()) {
        auto const &output = output_pair.second;
        callopts.add_fetch(output.name());
        auto const &outtensor = outputs_.emplace_back(output.dtype(), output.tensor_shape());
        outputdata_.emplace_back(outtensor.data());
    }

    auto const statusc = session_->MakeCallable(callopts, &callable_);

    if (!statusc.ok()) {
        throw std::runtime_error("failed to create callable handle from session");
    }

}

narf::tf_model_data::tf_model_data(tf_model_data&&) = default;

narf::tf_model_data::~tf_model_data() {
    if (session_) {
        auto const status = session_->ReleaseCallable(callable_);
        auto const statusc = session_->Close();
        // intentionally don't check the status here since we don't really care if it fails
    }
}

narf::tf_helper::tf_helper(const std::string &model_dir, const std::string &sig_name, const unsigned int nslots) {

    const unsigned int nslots_actual = std::max(nslots, 1U);

    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

    session_options.config.set_inter_op_parallelism_threads(-1);
    session_options.config.set_intra_op_parallelism_threads(1);

    // // load prototype model
    // auto const status = tensorflow::LoadSavedModel(session_options,
    //                                     run_options,
    //                                     model_dir,
    //                                     {tensorflow::kSavedModelTagServe},
    //                                     model_.get());
    //
    // auto const &signature = model_->GetSignatures().at(sig_name);
    //
    // tensorflow::CallableOptions callopts;
    //
    // for (auto const &input_pair : signature.inputs()) {
    //     auto const &input = input_pair.second;
    //     callopts.add_feed(input.name());
    // }
    //
    // for (auto const &output_pair : signature.outputs()) {
    //     auto const &output = output_pair.second;
    //     callopts.add_fetch(output.name());
    // }
    //
    // auto const statusc = model_->GetSession()->MakeCallable(callopts, &callable_);
    //
    // if (!statusc.ok()) {
    //     throw std::runtime_error("failed to create callable handle from session");
    // }

    // now construct actual sessions without reading again from disk
    model_data_.reserve(nslots_actual);
    for (unsigned int islot = 0; islot < nslots_actual; ++islot) {
        std::cout << "init model_data " << islot << std::endl;
        // model_data_.emplace_back(model_.meta_graph_def, model_dir, sig_name);
        model_data_.emplace_back(model_dir, sig_name);
        // model_data_.emplace_back(*model_, callable_, sig_name);
    }
}

// narf::tf_helper::~tf_helper() {
//    if (model_) {
//         auto const status = model_->GetSession()->ReleaseCallable(callable_);
//    }
// }


narf::tf_helper_pll::tf_helper_pll(const std::string &model_dir, const std::string &sig_name) :
    model_(std::make_unique<tensorflow::SavedModelBundleLite>()) {


    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

    session_options.config.set_inter_op_parallelism_threads(-1);
    session_options.config.set_intra_op_parallelism_threads(1);

    auto const status = tensorflow::LoadSavedModel(session_options,
                                        run_options,
                                        model_dir,
                                        {tensorflow::kSavedModelTagServe},
                                        model_.get());

    if (!status.ok()) {
       throw std::runtime_error("failed to load saved model from directory");
    }

    session_ = model_->GetSession();

    auto const &signature = model_->GetSignatures().at(sig_name);

    tensorflow::CallableOptions callopts;

    inputs_.reserve(signature.inputs().size());
    inputdata_.reserve(signature.inputs().size());
    for (auto const &input_pair : signature.inputs()) {
        auto const &input = input_pair.second;
        callopts.add_feed(input.name());
        auto &intensor = inputs_.emplace_back(input.dtype(), input.tensor_shape());
        inputdata_.emplace_back(intensor.data());
    }

    outputs_.reserve(signature.outputs().size());
    outputdata_.reserve(signature.outputs().size());
    for (auto const &output_pair : signature.outputs()) {
        auto const &output = output_pair.second;
        callopts.add_fetch(output.name());
        auto const &outtensor = outputs_.emplace_back(output.dtype(), output.tensor_shape());
        outputdata_.emplace_back(outtensor.data());
    }

    auto const statusc = model_->GetSession()->MakeCallable(callopts, &callable_);

    if (!statusc.ok()) {
        throw std::runtime_error("failed to create callable handle from session");
    }

}

narf::tf_helper_pll::tf_helper_pll(tf_helper_pll&&) = default;

// narf::tf_helper_pll::~tf_helper_pll() = default;

narf::tf_helper_pll::~tf_helper_pll() {
    if (session_) {
        auto const status = session_->ReleaseCallable(callable_);
        // intentionally don't check the status here since we don't really care if it fails
    }
}

void narf::tf_helper_pll::InvokeModel() {
    auto const status = session_->RunCallable(callable_, inputs_, &outputs_, nullptr);

    if (!status.ok()) {
        throw std::runtime_error("failed to invoke model");
    }
}

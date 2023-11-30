#ifndef NARF_TFCUTILS_H
#define NARF_TFCUTILS_H

#include "tensorflow/c/c_api.h"
// #include "tensorflow/cc/client/client_session.h"


namespace narf {

//     class tf_graph {
//     public:
//         tf_graph() : graph_(TF_NewGraph()) {}
//         ~tf_graph() { TF_DeleteGraph(graph_); }
//
//         TF_Graph *data() { return graph_; }
//
//     private:
//         TF_Graph *graph_;
//     };
//
//     class tf_sessionoptions {
//     public:
//         tf_sessionoptions() : sessionOptions_(TF_NewSessionOptions()) {}
//         ~tf_sessionoptions() { TF_DeleteSessionOptions(sessionOptions_); }
//
//          TF_Graph *data() { return graph_; }
//
//
//     private:
//         TF_SessionOptions *sessionOptions_;
//     };
//
//     class tf_status {
//     public:
//         tf_status() : status_(TF_NewStatus()) {}
//         ~tf_status() { TF_DeleteStatus(status_); }
//
//     private:
//         TF_Status *status_;
//     };


    class tf_session {
    public:
        tf_session(const char *filename, const char *signature = "serving_default_args_0", const char *tag = "serve") :
            graph_(TF_NewGraph()),
            sessionopts_(TF_NewSessionOptions()),
            status_(TF_NewStatus()) {

            uint8_t intra_op_parallelism_threads = 1;
            uint8_t inter_op_parallelism_threads = -1;
            uint8_t buf[] = {0x10, intra_op_parallelism_threads, 0x28, inter_op_parallelism_threads};
            TF_SetConfig(sessionopts_, buf, sizeof(buf), status_);

            // const char *tags = tag.c_str();
            // TF_Buffer *runopts = nullptr;
            session_ = TF_LoadSessionFromSavedModel(sessionopts_, runopts_, filename, &tag, 1, graph_, nullptr, status_);
            if (TF_GetCode(status_) != TF_OK) {
                throw std::runtime_error("Failed to load session from saved model");
            }


            TF_Operation *op = TF_GraphOperationByName(graph_, signature);

            if (TF_GetCode(status_) != TF_OK) {
                throw std::runtime_error("Failed to load operation from graph");
            }

            std::cout << "op " << op << std::endl;

            const int ninputs = TF_OperationNumInputs(op);
            const int noutputs = TF_OperationNumOutputs(op);

            std::cout << ninputs << " " << noutputs << std::endl;

        }

        ~tf_session() {
            if (session_) TF_DeleteSession(session_, status_);
            TF_DeleteStatus(status_);
            TF_DeleteSessionOptions(sessionopts_);
            TF_DeleteGraph(graph_);
        }


    private:
        TF_Buffer *runopts_ = nullptr;
        TF_Graph *graph_;
        TF_SessionOptions *sessionopts_;
        TF_Status *status_;
        TF_Session *session_ = nullptr;
    };

}

#endif

import tensorflow as tf

def function_to_tflite(funcs, input_signatures):
    """Convert function to tflite model using python dynamic execution trickery to ensure that inputs
    and outputs are alphabetically ordered, since this is apparently the only way to prevent tflite from
    scrambling them"""

    if not isinstance(funcs, list):
        funcs = [funcs]
        input_signatures = [input_signatures]

    def wrapped_func(iif, *args):
        outputs = funcs[iif](*args)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        output_dict = {}
        for i,output in enumerate(outputs):
            output_name = f"output_{iif:05d}_{i:05d}"
            output_dict[output_name] = output

        return output_dict

    arg_string = []
    for iif, input_signature in enumerate(input_signatures):
        inputs = []
        for i in range(len(input_signature)):
            input_name = f"input_{iif:05d}_{i:05d}"
            inputs.append(input_name)
        arg_string.append(", ".join(inputs))

    def_string = ""
    def_string += "def make_module(wrapped_func, input_signatures):\n"
    def_string += "    class Export_Module(tf.Module):\n"
    for i, func in enumerate(funcs):
        def_string += f"        @tf.function(input_signature = input_signatures[{i}])\n"
        def_string += f"        def {func.__name__}(self, {arg_string[i]}):\n"
        def_string += f"            return wrapped_func({i}, {arg_string[i]})\n"
    def_string += "    return Export_Module"

    ldict = {}
    exec(def_string, globals(), ldict)

    make_module = ldict["make_module"]
    Export_Module = make_module(wrapped_func, input_signatures)

    module = Export_Module()
    concrete_functions = [getattr(module, func_name).get_concrete_function() for func_name in [f.__name__ for f in funcs]]
    converter = tf.lite.TFLiteConverter.from_concrete_functions(concrete_functions, module)

    # enable TenorFlow ops and DISABLE builtin TFLite ops since these apparently slow things down
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    converter._experimental_allow_all_select_tf_ops = True

    tflite_model = converter.convert()

    test_interp = tf.lite.Interpreter(model_content = tflite_model)
    print(test_interp.get_input_details())
    print(test_interp.get_output_details())
    print(test_interp.get_signature_list())

    return tflite_model



def function_to_saved_model(func, input_signature, output):

    class Export_Module(tf.Module):
        @tf.function(input_signature = input_signature)
        def __call__(self, *args):
            return func(*args)

    model = Export_Module()

    tf.saved_model.save(model, output)

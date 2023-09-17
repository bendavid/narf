import tensorflow as tf

def function_to_tflite(func, input_signature):
    """Convert function to tflite model using python dynamic execution trickery to ensure that inputs
    and outputs are alphabetically ordered, since this is apparently the only way to prevent tflite from
    scrambling them"""

    inputs = []
    for i in range(len(input_signature)):
        input_name = f"input_{i:05d}"
        inputs.append(input_name)

    def wrapped_func(*args):
        outputs = func(*args)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        output_dict = {}
        for i,output in enumerate(outputs):
            output_name = f"output_{i:05d}"
            output_dict[output_name] = output

        return output_dict

    arg_string = ", ".join(inputs)

    def_string = ""
    def_string += "def make_module(wrapped_func, input_signature):\n"
    def_string += "    class Export_Module(tf.Module):\n"
    def_string += "        @tf.function(input_signature = input_signature)\n"
    def_string += f"        def __call__(self, {arg_string}):\n"
    def_string += f"            return wrapped_func({arg_string})\n"
    def_string += "    return Export_Module"

    ldict = {}
    exec(def_string, globals(), ldict)

    make_module = ldict["make_module"]
    Export_Module = make_module(wrapped_func, input_signature)

    module = Export_Module()
    concrete_function = module.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function], module)

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

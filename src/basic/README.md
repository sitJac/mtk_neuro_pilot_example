# A Generic Classifier to Support Classification Application

## What's in the example
* This example implements a digit recognition application using a CIFAR-10 TFLite model.

## What you'll learn
* How to deploy a trained TFLite model to an edge device using NeuroPilot SDK APIs.

## NeuroPilot SDK API quick introduction
### This is a brief introduction to the basic API of NeuroPilot SDK, please refer to the header file for more details.
* Developers can get a TensorFlow Lite (TFLite) model from TensorFlow model by using TFLite Toco convertor. NeuroPilot provides an extended API to interpreter TFLite model directly on the fly. This API will create an ANeuralNetworksTFLite instance and need a specific TFLite model path.
    ```
    int ANeuroPilotTFLiteWrapper_makeTFLite(ANeuralNetworksTFLite** tflite, const char* modelPath);
    ```
* You can also load the model from memory. There are two application scenarios in this part. One is used to accelerate when memory is loaded when it is used repeatedly. The other is that if your model has security consideration, you can decrypt and load it through this way.
    ```
    int ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize);
    ```

* Get the number of input/output tensors associated with the model.
    ```
    typedef enum {
        TFLITE_BUFFER_TYPE_INPUT = 0,
        TFLITE_BUFFER_TYPE_OUTPUT = 1,
    } NpTFLiteBufferType;

    int ANeuroPilotTFLiteWrapper_getTensorCount(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype, int32_t * count)
    ```

* Get the Rank information of the input/output tensor with the given index.
    ```
    int ANeuroPilotTFLiteWrapper_getTensorDimensions(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype, int index, int* dimensions)
    ```

* Get the size of input/output tensor with the given index in bytes.
    ```
    int ANeuroPilotTFLiteWrapper_getTensorByteSize(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype, int index, size_t* size)
    ```

* Get the data type information of the input/output tensor with the given index.
    ```
    typedef enum {
        TFLITE_TENSOR_TYPE_NONE = 0,
        TFLITE_TENSOR_TYPE_FLOAT = 1,
        TFLITE_TENSOR_TYPE_UINT8 = 2,
    } NpTFLiteTensorType;

    int ANeuroPilotTFLiteWrapper_getTensorType(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype, int index, TFLiteTensorType* ttype)
    ```

* Copies from the provided input buffer into the input tensor's buffer.
    ```
    int ANeuroPilotTFLiteWrapper_setInputTensorData(ANeuralNetworksTFLite* tflite, int index, const void* data, size_t size)
    ```

* Copies to the provided output buffer from the output tensor's buffer.
    ```
    int ANeuroPilotTFLiteWrapper_getOutputTensorData(ANeuralNetworksTFLite* tflite, int index, void* data, size_t size)
    ```

* Start inference by using ANeuroPilotTFLiteWrapper_invoke. Once finished, you can get the output data by using ANeuroPilotTFLiteWrapper_getTensor and ANeuroPilotTFLiteWrapper_getTensorByIndex API.
    ```
    int ANeuroPilotTFLiteWrapper_invoke(ANeuralNetworksTFLite* tflite);
    ```

* Store dequantized contents of the given output tensor to user-allocated buffer. This function is only used with quantized model.
    ```
    int ANeuroPilotTFLiteWrapper_getDequantizedOutputByIndex(ANeuralNetworksTFLite* tflite, void* buffer, size_t bufferByteSize, int tensorIndex);
    ```

* Free ANeuralNetworksTFLite instance.
    ```
    void ANeuroPilotTFLiteWrapper_free(ANeuralNetworksTFLite* tflite);
    ```

## NeuroPilot SDK API tutorial
### This tutorial is pseudocode sample, more completed usage is wrote in sample code
1. Include headers, including standard Android Neural Network and Mediatek proprietary API.
    ```
    #include <android/NeuralNetworks.h>
    #include "NeuroPilotTFLiteShim.h"
    ```

2. Specify the paths of TFLite model and input.
    ```
    const char* model_path = "mobilenet.tflite";
    const char* input_path = "mobilenet_input.bin";
    ```

3. Create ANerualNetworksTFLite instance.
    ```
    ANeuralNetworksTFLite* tflite = nullptr;
    if (ANeuroPilotTFLiteWrapper_makeTFLite(&tflite,
                                            model_path) != ANEURALNETWORKS_NO_ERROR) {
    }
    ```

4. Set input tensor data.
    ```
    if (ANeuroPilotTFLiteWrapper_setInputTensorData(tflite,
                                                    input_index,
                                                    input_buffer,
                                                    input_buffer_size) != ANEURALNETWORKS_NO_ERROR) {
        ANeuroPilotTFLiteWrapper_free(tflite);
    }
    ```

5. Start to invoke.
    ```
    if (ANeuroPilotTFLiteWrapper_invoke(tflite) != ANEURALNETWORKS_NO_ERROR) {
        ANeuroPilotTFLiteWrapper_free(tflite);
    }
    ```

6. Get output tensor data.
    ```
    if (ANeuroPilotTFLiteWrapper_getOutputTensorData(tflite,
                                                    output_index,
                                                    output_buffer,
                                                    output_buffer_size) != ANEURALNETWORKS_NO_ERROR) {
        ANeuroPilotTFLiteWrapper_free(tflite);
    }
    ```

7. Prepare the float buffer to store the dequantized output if the ANeuralNetworksTFLite instance is created from a quantized model.
    ```
    int bufferSize = output_buffer_size * sizeof(float);
    float buffer[bufferSize];
    ```

8. Get the dequantized output if the ANeuralNetworksTFLite instance is created from a quantized model.
    ```
    if (ANeuroPilotTFLiteWrapper_getDequantizedOutputByIndex(tflite,
                                                            (void*)buffer,
                                                            bufferSize,
                                                            0) != ANEURALNETWORKS_NO_ERROR) {
        ANeuroPilotTFLiteWrapper_free(tflite);
    }
    ```

9. Free the ANeuralNetworksTFLite instance before ending the program.
    ```
    ANeuroPilotTFLiteWrapper_free(tflite);
    ```

## Source code module description
* GenericClassifier.cpp & GenericClassifier.h
    * A C++ class that implements the on-device inference functionality via the NeuroPilot API.
* Main.cpp
    * The application main entry, which handles the user's command line parameters.
    * User need to specify the TFLite model, input bin file by command line parameters.
    * The main entry will create a GenericClassifier object for on-device inference.

## Application execution flow
1. User needs to specify the TFLite model, input bin file by command line parameters.
    * tflite_model, input, output are requirem parameters.
2. In application main entry, the parameters will be processed in *process_options()*.
3. Construct a GenericClassifier instance with user sprcified TFLite model.
    * GenericClassifier owns the ANeuralNetworksTFLite handle that represents the TFLite model context created from *ANeuroPilotTFLiteWrapper_makeTFLite()*.
    * The developers also can use the following APIs to get TFLite model information.
        * *ANeuroPilotTFLiteWrapper_getTensorByteSize()* : Get the size of the underlying data in bytes.
        * *ANeuroPilotTFLiteWrapper_getTensorType()* : Get the data type information of the input/output tensor with the given index.
    * Print the model information if necessary.
        * If user specifies model_info in the command line arguments, print the following model information.
            * Input tensor data type.
            * Output tensor data type.
            * Input tensor rank and dimensions.
            * Output tensor rank and dimensions.
    * Set softmax flag if necessary.
        * If user specifies enable_softmax in the command line arguments, enable the mApplySoftmax flag in GenericClassifier.
4. Perform on-device inference with user specified input data. And get the inference results.
    * For image classification, the input data must be the image's raw pixel data. (RGB values within [0, 255])
        * The developers are responsible for the pre-processing of the input images.
            * Resize the input image to fit the input tensor dimensions of the model.
            * Convert the resized image to raw pixel data.
    * GenericClassifier uses *ANeuroPilotTFLiteWrapper_setInputTensorData()* to set the read input data into the TFLite model context.
        * For multiple input tensors, the corresponding input data can be set via *ANeuroPilotTFLiteWrapper_setInputTensorData()* with spcified tensor indexes.
    * Next, the GenericClassifier will use *ANeuroPilotTFLiteWrapper_invoke()* to trigger on-device inference.
    * When *ANeuroPilotTFLiteWrapper_invoke()* is complete, it can be used via the *ANeuroPilotTFLiteWrapper_getOutputTensorData()* to retrieve inference results.
        * For quantized model, the developers may need a post-processing to dequantize the inference results. Since the quantization parameters are defined in the TFLite model, NeuroPilot provides an API for the user to obtain the dequantized inference result. As a result, developers do not need to implement their own post-processing of dequantization. Use *ANeuroPilotTFLiteWrapper_getDequantizedOutputByIndex()* to get the dequantized inference results.
        * For float model, use *ANeuroPilotTFLiteWrapper_getOutputTensorData()* to get the float inferenceresults directly.
    5. In GenericClassifier's destructor, the TFLite model context created from *ANeuroPilotTFLiteWrapper_makeTFLite()* should be released via *ANeuroPilotTFLiteWrapper_free()*.

```
                                                     Generic Classifier - Sequence Diagram

                    ┌────┐                   ┌─────────────────┐                                                ┌──────────┐
                    │Main│                   │GenericClassifier│                                                │NeuroPilot│
                    └─┬──┘                   └────────┬────────┘                                                └────┬─────┘
       main()        ┌┴┐                              │                                                              │
 ───────────────────>│ │     ────┐                    │                                                              │
                     │ │         │ process_options()  │                                                              │
                     │ │     <───┘                    │                                                              │
                     │ │                              │                                                              │
                     │ │                              │                                                              │
                     │ │                              │                                                              │
                     │ │                              │                                                              │
                     │ │─ ─ ┐                         │                                                              │
                     │ │    |                         │                                                              │
                     │ │< ─ ┘                         │                                                              │
                     │ ┌┴┐Construct GenericClassifier┌┴┐                                                             │
                     │ │ │ ─────────────────────────>│ │     ────┐                                                   │
                     │ │ │                           │ │         │ GenericClassifier()                               │
                     │ │ │                           │ │     <───┘                                                   │
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │           ANeuroPilotTFLiteWrapper_makeTFLite()             ┌┴┐
                     │ │ │                           │ │ ──────────────────────────────────────────────────────────> │ │
                     │ │ │                           │ │                                                             └┬┘
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ ┌┴<─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │       ANeuroPilotTFLiteWrapper_getTensorByteSize()        ┌┴┐
                     │ │ │                           │ │ │ ─────────────────────────────────────────────────────────>│ │
                     │ │ │                           │ │ │                                                           └┬┘
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │       ANeuroPilotTFLiteWrapper_getTensorByteSize()        ┌┴┐
                     │ │ │                           │ │ │ ─────────────────────────────────────────────────────────>│ │
                     │ │ │                           │ │ │                                                           └┬┘
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ └┬┘ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │          ANeuroPilotTFLiteWrapper_getTensorType()           ┌┴┐
                     │ │ │                           │ │ ──────────────────────────────────────────────────────────> │ │
                     │ │ │                           │ │                                                             └┬┘
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │─ ─ ┐                                                        │
                     │ │ │                           │ │    |                                                        │
                     │ │ │                           └┬┘< ─ ┘                                                        │
                     │ │ │                            │                                                              │
                     │ │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                                                              │
                     │ │ │                            │                                                              │
                     │ │ │                            │                                                              │
          ╔══════╤═══╪═╪═╪════════════════════════════╪══════════════════╗                                           │
          ║ ALT  │  print model information           │                  ║                                           │
          ╟──────┘   │ │ │                            │                  ║                                           │
          ║          │ │ │  PrintModelInformation()  ┌┴┐                 ║                                           │
          ║          │ │ │ ─────────────────────────>│ │                 ║                                           │
          ║          │ │ │                           └┬┘                 ║                                           │
          ║          │ │ │                            │                  ║                                           │
          ║          │ │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                  ║                                           │
          ╚══════════╪═╪═╪════════════════════════════╪══════════════════╝                                           │
                     │ │ │                            │                                                              │
                     │ │ │                            │                                                              │
          ╔══════╤═══╪═╪═╪════════════════════════════╪══════════════════╗                                           │
          ║ ALT  │  enable softmax                    │                  ║                                           │
          ╟──────┘   │ │ │                            │                  ║                                           │
          ║          │ │ │      EnableSoftmax()      ┌┴┐                 ║                                           │
          ║          │ │ │ ─────────────────────────>│ │                 ║                                           │
          ║          │ │ │                           └┬┘                 ║                                           │
          ║          │ │ │                            │                  ║                                           │
          ║          │ │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                  ║                                           │
          ╚══════════╪═╪═╪════════════════════════════╪══════════════════╝                                           │
                     │ │ │                            │                                                              │
                     │ │ │        Inference()        ┌┴┐                                                             │
                     │ │ │ ─────────────────────────>│ │     ────┐                                                   │
                     │ │ │                           │ │         │ ReadInput()                                       │
                     │ │ │                           │ │     <───┘                                                   │
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │                                                             │
                     │ │ │                           │ │─ ─ ┐                                                        │
                     │ │ │                           │ │    |                                                        │
                     │ │ │                           │ │< ─ ┘                                                        │
                     │ │ │                           │ ┌┴┐       ANeuroPilotTFLiteWrapper_setInputTensorData()       ┌┴┐
                     │ │ │                           │ │ │ ─────────────────────────────────────────────────────────>│ │
                     │ │ │                           │ │ │                                                           └┬┘
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │             ANeuroPilotTFLiteWrapper_invoke()             ┌┴┐
                     │ │ │                           │ │ │ ─────────────────────────────────────────────────────────>│ │
                     │ │ │                           │ │ │                                                           └┬┘
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │      ANeuroPilotTFLiteWrapper_getOutputTensorData()       ┌┴┐
                     │ │ │                           │ │ │ ─────────────────────────────────────────────────────────>│ │
                     │ │ │                           │ │ │                                                           └┬┘
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │ <─ ─────┐              ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                     │ │ │                           │ │ │         │ SaveOutput()                                    │
                     │ │ │                           │ │ │     <───┘                                                 │
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │                                                           │
                     │ │ │                           │ │ │─ ─ ┐                                                      │
                     │ │ │                           │ │ │    | ────┐                                                │
                     │ │ │                           │ │ │< ─ ┘     │ GetPredictedClass()                            │
                     │ │ │                           │ │┌┴┐     <───┘                                                │
                     │ │ │                           │ ││ │                                                          │
                     │ │ │                           │ ││ │     ────┐                                                │
                     │ │ │                           │ ││ │         │ DequantizeOutput()                             │
                     │ │ │                           │ ││ │     <───┘                                                │
                     │ │ │                           │ ││ │                                                          │
                     │ │ │                           │ ││ │                                                          │
                     │ │ │                           │ ││ │                                                          │
                     │ │ │                           │ ││ ┌┴┐                                                        │
                     │ │ │                           │ ││ │ │─ ─ ┐                                                   │
                     │ │ │                           │ ││ │ │    |                                                   │
                     │ │ │                           │ ││ │ │< ─ ┘                                                   │
                     │ │ │                           │ ││ │┌┴┐                                                       │
                     │ │ │         ╔══════╤══════════╪═╪╪═╪╪═╪═══════════════════════════════════════════════════════╪═══════════════╗
                     │ │ │         ║ ALT  │  need to dequantize output                                               │               ║
                     │ │ │         ╟──────┘          │ ││ ││ │                                                       │               ║
                     │ │ │         ║                 │ ││ ││ │ANeuroPilotTFLiteWrapper_getDequantizedOutputByIndex() ┌┴┐             ║
                     │ │ │         ║                 │ ││ ││ │ ────────────────────────────────────────────────────> │ │             ║
                     │ │ │         ║                 │ ││ ││ │                                                       └┬┘             ║
                     │ │ │         ║                 │ ││ ││ │                                                       │               ║
                     │ │ │         ║                 │ ││ ││ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                ║
                     │ │ │         ╠═════════════════╪═╪╪═╪╪═╪═══════════════════════════════════════════════════════╪═══════════════╣
                     │ │ │         ║                 │ ││ │└┬┘                                                       │               ║
                     │ │ │         ║                 │ ││ │ │     ANeuroPilotTFLiteWrapper_getOutputTensorData()     ┌┴┐             ║
                     │ │ │         ║                 │ ││ │ │ ──────────────────────────────────────────────────────>│ │             ║
                     │ │ │         ║                 │ ││ │ │                                                        └┬┘             ║
                     │ │ │         ║                 │ ││ │ │                                                        │               ║
                     │ │ │         ║                 │ ││ │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─               ║
                     │ │ │         ╚═════════════════╪═╪╪═╪═╪════════════════════════════════════════════════════════╪═══════════════╝
                     │ │ │                           │ ││ │ │                                                        │
                     │ │ │                           │ ││ │ │                                                        │
                     │ │ │                           │ ││ │ │─ ─ ┐                                                   │
                     │ │ │                           │ ││ │ │    |                                                   │
                     │ │ │                           │ ││ └┬┘< ─ ┘                                                   │
                     │ │ │                           │ ││ │                                                          │
                     │ │ │ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │ │                                                          │
                     │ │ │                           │ ││ │                                                          │
                     │ │ │                           │ ││ │                                                          │
 <─ ─ ─ ─ ─ ─ ─ ─ ─ ─│ │ │                           │ ││ │                                                          │
                    ┌│ └┬┘                   ┌───────│ ││ │────┐                                                ┌────┴─────┐
                    ││ │n│                   │Generic│ ││ │fier│                                                │NeuroPilot│
                    └│ │─┘                   └───────│ ││ │────┘                                                └──────────┘
                     └┬┘                             └┬└└┬┘


```

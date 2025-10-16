# NeuroPilot Examples

Here are three examples using NeuroPilot APIs to do on-device inference.

# Package structure
```
|- basic
|   |- input
|   |- model
|- image_classification
|   |- input
|   |- model
|- object_detection
|   |- input
|   |- model
|- logger
|   |- include
|- profiler
|   |- include
|- utils
|   |- include
```

# Directory description
* basic
    * As a simple native application of the NeuroPilot APIs toturial, on-device inference can be performed using a classification model.
* image_classification
    * A native application for image classification.
* object_detection
    * A native application for object detection.
* logger
    * A tool that prints log information to the Android logging system and console.
* profiler
    * A tool that measures the time taken by a code portion.
* utils
    * File operation utilities.

# Setup the Android NDK toolchain
* Download & Unzip Android NDK r20
    ```
    wget https://dl.google.com/android/repository/android-ndk-r20-linux-x86_64.zip
    unzip android-ndk-r20-linux-x86_64.zip
    ```

# Build examples
* In CMakeLists.txt, change NDK_STANDALONE_TOOLCHAIN to correct path
* Make a build folder. And go to the build folder
    ```
    mkdir build
    cd build
    ```
* Setup 32-bit build if you need a 32-bit application
    ```
    cmake -DTARGET=arm ../
    ```
* Setup 64-bit build if you need a 64-bit application
    ```
    cmake -DTARGET=aarch64 ../
    ```
* Build
    ```
    make
    ```
* Check the output executable(GenericClassifier / ClassifyImage / DetectObject) in build folder.

# Basic(Generic classification)
* Parameters

    The binary takes the following required parameters:
    * `tflite_model` : `string` \
        Path to the TFlite model file.
    * `input` : `string` \
        Path to the input bin file.
    * `output` : `string` \
        Path to the output bin file.

    and the following optional parameters:
    * `model_info` : `int` (default=0) \
        Show the TFLite model information or not.
    * `enable_softmax` : `int` (default=0) \
        Apply softmax on the inference result or not.
* Run GenericClassifier
    * Prepare your test image bin file by the python script. And You will get a bin file converted from the image file.
        ```
        python process_input_image.py img_1.jpg 28 28
        ```
    * Connect your Android device. Push the binary to your Android device with adb push.
        ```
        adb push build\GenericClassifier /data/local/tmp/
        adb shell chmod +x /data/local/tmp/GenericClassifier
        adb push basic\input\img_1.jpg_2828.bin /data/local/tmp/
        adb push basic\model\mnist_uint8.tflite /data/local/tmp/
        ```
    * Run the binary.
        ```
        adb shell /data/local/tmp/GenericClassifier -i /data/local/tmp/img_1.jpg_2828.bin -o /data/local/tmp/img_1.jpg_output.bin -m /data/local/tmp/mnist_uint8.tflite -p 1
        ```
    * Check the inference output.
        ```
        [INFO][GenericClassifier]: Input tensor byte size: 784
        [INFO][GenericClassifier]: Output tensor byte size: 10
        [INFO][GenericClassifier]: Input tensor data type: uint8
        [INFO][GenericClassifier]: Output tensor data type: uint8
        [INFO][GenericClassifier]: Input tensor rank: 4, dimensions: [1 ,28 ,28 ,1]
        [INFO][GenericClassifier]: Output tensor rank: 2, dimensions: [1 ,10]
        [INFO][GenericClassifier]: Probility[0] = -12.088
        [INFO][GenericClassifier]: Probility[1] = -19.946
        [INFO][GenericClassifier]: Probility[2] = 40.496
        [INFO][GenericClassifier]: Probility[3] = 4.835
        [INFO][GenericClassifier]: Probility[4] = 0.000
        [INFO][GenericClassifier]: Probility[5] = -32.638
        [INFO][GenericClassifier]: Probility[6] = -30.825
        [INFO][GenericClassifier]: Probility[7] = -4.231
        [INFO][GenericClassifier]: Probility[8] = -7.253
        [INFO][GenericClassifier]: Probility[9] = -9.671
        [INFO][GenericClassifierApp]: Predicted digit: 2
        ```

# Image classification
* Parameters

    The binary takes the following required parameters:
    * `image_bin` : `string` \
        Path to the input bin file converted from image(bmp/jpg).
    * `labels` : `string` \
        Path to the labels text file.
    * `tflite_model` : `string` \
        Path to the TFlite model file.

    and the following optional parameters:
    * `allow_fp16` : `int` (default=0) \
        Allow running float models with FP16 precision not.
    * `count` : `int` (default=1) \
        Invoke the inference for certain loops.
    * `input_mean` : `float` (default=127.5) \
        Input mean.
    * `input_std` : `float` (default=127.5) \
        Input standard deviation
    * `profiling` : `int` (default=0) \
        Show profiling summary or not.
    * `num_results` : `int` (default=5) \
        Number of results to show.
* Run ClassifyImage
    * Prepare your test image bin file by the python script. And You will get a bin file converted from the image file. BMP and JPG files are acceptable. Resize the image to width:224, height:224, and convert the resized image into a byte array with dimension: [1, 224, 224, 3]
        ```
        python process_input_image.py grace_hopper.bmp 224 224
        ```
    * Prepare your TFLite model file. Or use the existing TFLite models in the model folder.
    * Connect your Android device. Push the binary to your Android device with adb push.
        ```
        adb push build\ClassifyImage /data/local/tmp/
        adb shell chmod +x /data/local/tmp/ClassifyImage
        adb push image_classification\input\grace_hopper.bmp_224224.bin /data/local/tmp/
        adb push image_classification\model\labels_imagenet_slim.txt /data/local/tmp/
        adb push image_classification\model\mobilenet_v1_1.0_224.tflite /data/local/tmp/
        adb push image_classification\model\mobilenet_v1_1.0_224_quant.tflite /data/local/tmp/
        ```
    * Run the binary.
        ```
        adb shell /data/local/tmp/ClassifyImage -c 10 -p 1 -v 1 -m /data/local/tmp/mobilenet_v1_1.0_224_quant.tflite -l /data/local/tmp/labels_imagenet_slim.txt -i /data/local/tmp/grace_hopper.bmp_224224.bin
        ```
    * Check the inference output.
        ```
        [INFO][MainApp]: Construct engine with model
        [INFO][MainApp]: Inference with classification engine
        [VERB][Engine]: Confidence: 0.917647, index:653, label:military uniform
        [VERB][Engine]: Confidence: 0.015686, index:907, label:Windsor tie
        [VERB][Engine]: Confidence: 0.007843, index:668, label:mortarboard
        [VERB][Engine]: Confidence: 0.007843, index:466, label:bulletproof vest
        [VERB][Engine]: Confidence: 0.003922, index:820, label:stage
        [INFO][MainApp]: Top-5 results
        [INFO][MainApp]:     military uniform
        [INFO][MainApp]:     Windsor tie
        [INFO][MainApp]:     mortarboard
        [INFO][MainApp]:     bulletproof vest
        [INFO][MainApp]:     stage

        Profiling Summary:
        -------------------------------------------------------------------------------------------------
                    Function     Calls     Total time  Avg time  Min time  Max time
        -------------------------------------------------------------------------------------------------
            Construct engine         1       274.88ms  274.88ms  274.88ms  274.88ms
                  Copy input         1       155.23us  155.23us  155.23us  155.23us
                      Invoke        10       268.37ms   26.84ms   24.70ms   34.12ms
                 Copy output         1        62.77us   62.77us   62.77us   62.77us
        ```

# Object detection
* Parameters

    The binary takes the following required parameters:
    * `image_bin` : `string` \
        Path to the input bin file converted from image(bmp/jpg).
    * `labels` : `string` \
        Path to the labels text file.
    * `tflite_model` : `string` \
        Path to the TFlite model file.

    and the following optional parameters:
    * `allow_fp16` : `int` (default=0) \
        Allow running float models with FP16 precision not.
    * `count` : `int` (default=1) \
        Invoke the inference for certain loops.
    * `input_mean` : `float` (default=127.5) \
        Input mean.
    * `input_std` : `float` (default=127.5) \
        Input standard deviation
    * `profiling` : `int` (default=0) \
        Show profiling summary or not.
    * `num_results` : `int` (default=5) \
        Number of results to show.
* Run DetectObject
    * Prepare your test image bin file by the python script. And You will get a bin file converted from the image file. BMP and JPG files are acceptable. Resize the image to width:300, height:300, and convert the resized image into a byte array with dimension: [1, 300, 300, 3]
        ```
        python process_input_image.py voc_boat_pi.jpg 300 300
        ```
    * Prepare your TFLite model file. Or use the existing TFLite models in the model folder.
        ```
        http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
        ```
    * Connect your Android device. Push the binary to your Android device with adb push.
        ```
        adb push build\DetectObject /data/local/tmp/
        adb shell chmod +x /data/local/tmp/DetectObject
        adb push object_detection\input\voc_boat_pi.jpg_300300.bin /data/local/tmp/
        adb push object_detection\model\labelmap.txt /data/local/tmp/
        adb push object_detection\model\detect.tflite /data/local/tmp/
        ```
    * Run the binary.
        ```
        adb shell /data/local/tmp/DetectObject -c 10 -p 1 -v 1 -m /data/local/tmp/detect.tflite -l /data/local/tmp/labelmap.txt -i /data/local/tmp/voc_boat_pi.jpg_300300.bin
        ```
    * Check the inference output.
        ```
        [INFO][MainApp]: Construct engine with model
        [INFO][MainApp]: Inference with detection engine
        [VERB][Engine]: Confidence: 0.710938, index:18, label:dog, left:25.970484, top:86.042816, right:119.217453, bottom:272.027649
        [VERB][Engine]: Confidence: 0.687500, index:9, label:boat, left:4.823184, top:140.229782, right:247.917755, bottom:292.747314
        [VERB][Engine]: Confidence: 0.523438, index:1, label:person, left:157.267212, top:44.324677, right:187.827652, bottom:187.827591
        [VERB][Engine]: Confidence: 0.390625, index:18, label:dog, left:14.383319, top:119.849197, right:171.799103, bottom:287.538513
        [VERB][Engine]: Confidence: 0.390625, index:49, label:knife, left:134.806290, top:205.548660, right:189.451141, bottom:218.517456
        [VERB][Engine]: Confidence: 0.332031, index:61, label:cake, left:78.390984, top:155.650711, right:251.008881, bottom:285.526703
        [VERB][Engine]: Confidence: 0.332031, index:49, label:knife, left:102.349579, top:194.264877, right:193.089157, bottom:228.944473
        [VERB][Engine]: Confidence: 0.312500, index:77, label:cell phone, left:174.874603, top:63.379471, right:179.898529, bottom:75.747704
        [VERB][Engine]: Confidence: 0.312500, index:40, label:baseball glove, left:64.299103, top:96.999496, right:122.723625, bottom:178.235168
        [VERB][Engine]: Confidence: 0.289062, index:55, label:orange, left:-0.246850, top:189.475937, right:37.284611, bottom:263.361786
        [INFO][MainApp]: Detections:
        [INFO][MainApp]:     dog
        [INFO][MainApp]:     - confidence: 0.710938
        [INFO][MainApp]:     - rectangular window(left/top/right/bottom):25.970484/86.042816/119.217453/272.027649
        [INFO][MainApp]:     boat
        [INFO][MainApp]:     - confidence: 0.687500
        [INFO][MainApp]:     - rectangular window(left/top/right/bottom):4.823184/140.229782/247.917755/292.747314
        [INFO][MainApp]:     person
        [INFO][MainApp]:     - confidence: 0.523438
        [INFO][MainApp]:     - rectangular window(left/top/right/bottom):157.267212/44.324677/187.827652/187.827591

        Profiling Summary:
        -------------------------------------------------------------------------------------------------
                    Function     Calls     Total time  Avg time  Min time  Max time
        -------------------------------------------------------------------------------------------------
            Construct engine         1       478.36ms  478.36ms  478.36ms  478.36ms
                  Copy input         1       278.00us  278.00us  278.00us  278.00us
                      Invoke        10       654.44ms   65.44ms   60.02ms   71.97ms
           Copy output boxes         1        68.85us   68.85us   68.85us   68.85us
         Copy output classes         1         2.46us    2.46us    2.46us    2.46us
          Copy output scores         1         2.08us    2.08us    2.08us    2.08us
      Copy output detections         1         2.00us    2.00us    2.00us    2.00us
        ```
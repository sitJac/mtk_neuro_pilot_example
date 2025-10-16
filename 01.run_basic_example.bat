
adb wait-for-device
adb root
adb remount

adb push ./src/basic/model/mnist_uint8.tflite /data/local/tmp/
adb push ./src/basic/input/img_1.jpg_2828.bin /data/local/tmp/

adb shell /data/local/tmp/GenericClassifier ^
 -m /data/local/tmp/mnist_uint8.tflite ^
 -i /data/local/tmp/img_1.jpg_2828.bin ^
 -o /data/local/tmp/output_img_1.jpg_2828.bin ^
 -p 1


adb wait-for-device
adb root
adb remount

adb push ..\src\image_classification\model\mobilenet_v2_1.0_224_quant.tflite /data/local/tmp/
adb push ..\src\image_classification\input\grace_hopper.bmp_224224.bin /data/local/tmp/
adb push ..\src\image_classification\model\labels_imagenet_slim.txt /data/local/tmp/

adb shell /data/local/tmp/ClassifyImage ^
 -m /data/local/tmp/mobilenet_v2_1.0_224_quant.tflite ^
 -i /data/local/tmp/grace_hopper.bmp_224224.bin ^
 -l /data/local/tmp/labels_imagenet_slim.txt ^
 -c 10 -p 1 -v 1

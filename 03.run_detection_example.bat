
adb wait-for-device
adb root
adb remount

adb push ./src/object_detection/model/detect.tflite /data/local/tmp/
adb push ./src/object_detection/input/voc_boat_pi.jpg_300300.bin /data/local/tmp/
adb push ./src/object_detection/model/labelmap.txt /data/local/tmp/

adb shell /data/local/tmp/DetectObject ^
 -m /data/local/tmp/detect.tflite  ^
 -i  /data/local/tmp/voc_boat_pi.jpg_300300.bin ^
 -l /data/local/tmp/labelmap.txt ^
 -c 10 -p 1 -v 1

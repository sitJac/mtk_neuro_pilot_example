## Installation

- Create virtual environment with specific Python version
`uv venv --python 3.8 neuropilot_env`

- Activate virtual environment
`source neuropilot_env/bin/activate && python –version`

- Install dependencies
`uv pip install -r requirements.txt`

- Install mtk-converter
`uv pip install mtk_converter-7.16.0+release-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl`
 
- Verify installation
`python3 -c 'import mtk_converter; print(mtk_converter.__version__)'`
> 7.16.0+release 


## Usage
- Command line example:
```bash
# Convert to TFLite format
(neuropilot_env) ➜  model_converter git:(master) ✗ mtk_caffe_converter      \    
    --input_prototxt_file=caffe/mnasnet_v3.prototxt                         \
    --input_caffemodel_file=caffe/mnasnet_v3_20250630.caffemodel            \
    --output_file=mnasnet_v3_20250630.tflite    

# Convert to MLIR format
(neuropilot_env) ➜  model_converter git:(master) ✗ mtk_caffe_converter      \    
    --input_prototxt_file=caffe/mnasnet_v3.prototxt                         \
    --input_caffemodel_file=caffe/mnasnet_v3_20250630.caffemodel            \
    --output_file_format=mlir                                               \
    --output_file=mnasnet_v3_20250630.mlir
```

- Python example:
```python
import mtk_converter
converter = mtk_converter.CaffeConverter.from_model_files(
    'caffe/mnasnet_v3.prototxt', 'caffe/mnasnet_v3_20250630.caffemodel'
)
# Convert to TFLite format
_ = converter.convert_to_tflite(output_file='mnasnet_v3_20250630.tflite')
# Convert to MLIR format
_ = converter.convert_to_mlir(output_file='mnasnet_v3_20250630.mlir')
```
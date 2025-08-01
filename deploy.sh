pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
pip config set global.trusted-host mirrors.cloud.tencent.com

pip install --upgrade pip

pip install "pydantic<2.0.0"
cd /path/to/LMM-Det
pip install -e .


pip install ninja
pip install flash-attn --no-build-isolation

pip install protobuf==3.20.0 
pip install openpyxl
# for vicuna 7B/13B
pip uninstall transformers -y
pip install transformers==4.38.2 transformers_stream_generator==0.0.5 tiktoken==0.6.0 
pip install accelerate==0.26.1
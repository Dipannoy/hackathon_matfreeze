#create virtual environment using uv

git clone https://github.com/Dipannoy/hackathon_matfreeze.git

uv venv --python 3.10 .venv

source /workspace/multimat_project/software/.venv/bin/activate

or if you are already in the folder do

source .venv/bin/activate

uv pip install -r requirements.txt

uv pip install "torch==2.2.2+cu118" -f https://download.pytorch.org/whl/cu118/torch_stable.html

uv pip install torch==2.2.2+cu118 torch-geometric==2.5.3 torch-scatter==2.1.2+pt22cu118 torch-sparse==0.6.18 pt22cu118 torch-cluster==1.6.3+pt22cu118 torch-spline-conv==1.2.2+pt22cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html -f https://data.pyg.org/whl/torch-2.2.2+cu118.html

uv pip install "pydantic==1.10.9" --force-reinstall

uv pip install einops --upgrade

Finall try with:

python multimat.py --data_path ./example_data --modalities_encoders crystal dos --exp 'pretrain'

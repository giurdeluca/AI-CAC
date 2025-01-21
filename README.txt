# Databricks notebook source
# MAGIC %md
# MAGIC Instructions for AI-CAC model inference on Non-gated Chest CT scans: 
# MAGIC
# MAGIC Please create a folder for each CT chest scan study and place DICOM files from that study within the folder. The folder name will be used as the name for that study. The code will take a root directory consisting of multiple folders from multiple studies for inference. The code will select a single non-contrast chest series per study that is most suitable for our CAC model. The software will create a table for reach dicom file from acceptable series and place them into a table with the following columns "StudyName", "DICOMFilePath", "". This table will be used by the inference code to run the model on each slice from the series and aggregate the results into a CAC score.
# MAGIC
# MAGIC Input: Specify root folder containing DICOM study subfolders
# MAGIC Output: Specify name/location of CSV file to be saved with AI generated CAC scores. 
# MAGIC Optional: Save PNG masks (alternative options coming soon)
# MAGIC
# MAGIC
# MAGIC  #96 is 100.4M params
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

Package Requirements:
    Databricks Runtime 11.3 LTS ML Environment g4dn.12xLarge
    WARNING: You are using pip version 21.2.4; however, version 24.0 is available. - message during HC2 training run 


    Python 3.9.19 (main, Apr  6 2024, 17:57:55) 
    pip 21.2.4 
Python Packages: 
    itk                                5.4.0
    matplotlib                         3.9.4
    monai                              1.4.0
    monailabel                         0.8.5
    nibabel                            5.3.2
    numpy                              1.26.4
    opencv-python                      4.11.0.86
    pandas                             1.3.4
    pillow                             11.1.0
    pydicom                            2.4.4
    python-gdcm                        3.0.24.1
    scikit-learn                       0.24.2
    scipy                              1.13.1
    SimpleITK                          2.4.1
    torch                              1.12.1+cu113
    torchmetrics                       1.5.2
    torchvision                        0.13.1+cu113


stats_init script:
    !/bin/bash
    pip install pydicom numpy scipy pandas scikit-learn matplotlib==3.4.3 lifelines==0.28.0
    pip install s3fs

gpu_init script:
    #!/bin/bash
    pip install torch torchvision torchmetrics monai[all]
    pip install pydicom numpy scipy pandas scikit-learn matplotlib
    pip install itk opencv-python pillow
    pip install simpleitk monailabel
    pip install s3fs

Prior v5 processing pydicom 2.3.1?

Public Init GPU script 
#!/bin/bash
pip install itk==5.4.0 matplotlib==3.9.4 monai[all]==1.4.0 monailabel==0.8.5 nibabel==5.3.2 
pip install numpy==1.26.4 opencv-python==4.11.0.86 pandas==1.3.4 pillow==11.1.0 pydicom==2.4.4 
pip install python-gdcm==3.0.24.1 scikit-learn==0.24.2 scipy==1.13.1 SimpleITK==2.4.1 
pip install torch==1.12.1+cu113 torchmetrics==1.5.2 torchvision==0.13.1+cu113 


# COMMAND ----------

1/17/25 GPU versions:
    Databricks Runtime 11.3 LTS ML Environment g4dn.12xLarge
    Python 3.9.19 (main, Apr  6 2024, 17:57:55) 
    pip 21.2.4 
pip list 
Package                            Version
---------------------------------- --------------------
absl-py                            1.0.0
aiobotocore                        2.18.0
aiohappyeyeballs                   2.4.4
aiohttp                            3.11.11
aioitertools                       0.12.0
aiosignal                          1.3.2
alembic                            1.14.0
annotated-types                    0.7.0
anyio                              4.8.0
argon2-cffi                        20.1.0
asciitree                          0.3.3
astor                              0.8.1
astunparse                         1.6.3
async-generator                    1.10
async-timeout                      5.0.1
attrs                              24.3.0
azure-core                         1.22.1
azure-cosmos                       4.2.0
backcall                           0.2.0
backports.entry-points-selectable  1.1.1
bcrypt                             4.2.1
beautifulsoup4                     4.12.3
black                              22.3.0
bleach                             4.0.0
blis                               0.7.8
boto3                              1.21.18
botocore                           1.36.1
cachetools                         5.5.0
catalogue                          2.0.8
certifi                            2021.10.8
cffi                               1.14.6
chardet                            4.0.0
charset-normalizer                 2.0.4
clearml                            1.17.0
click                              8.0.3
cloudpickle                        2.0.0
cmdstanpy                          0.9.68
colorama                           0.4.6
coloredlogs                        15.0.1
colorlog                           6.9.0
confection                         0.0.1
configparser                       5.2.0
contourpy                          1.3.0
convertdate                        2.4.0
cryptography                       3.4.8
cucim-cu12                         24.8.0
cupy-cuda12x                       13.3.0
cycler                             0.10.0
cymem                              2.0.6
Cython                             0.29.24
databricks-automl-runtime          0.2.11.1
databricks-cli                     0.17.3
databricks-sdk                     0.40.0
dbl-tempo                          0.1.12
dbus-python                        1.2.16
debugpy                            1.4.1
decorator                          5.1.0
defusedxml                         0.7.1
Deprecated                         1.2.15
dicomweb-client                    0.59.3
dill                               0.3.4
diskcache                          5.4.0
distlib                            0.3.6
distro                             1.4.0
distro-info                        0.23+ubuntu1.1
docker                             7.1.0
einops                             0.8.0
entrypoints                        0.3
ephem                              4.1.3
exceptiongroup                     1.2.2
expiring-dict                      1.1.1
expiringdict                       1.2.2
facets-overview                    1.0.0
fastapi                            0.115.6
fasteners                          0.19
fastrlock                          0.8.3
fasttext                           0.9.2
filelock                           3.16.1
fire                               0.7.0
Flask                              1.1.2+db1
flatbuffers                        1.12
fonttools                          4.55.3
frozenlist                         1.5.0
fsspec                             2024.12.0
furl                               2.1.3
future                             0.18.2
gast                               0.4.0
gdown                              5.2.0
girder-client                      3.2.6
gitdb                              4.0.9
GitPython                          3.1.27
google-auth                        2.37.0
google-auth-oauthlib               0.4.6
google-pasta                       0.2.0
graphene                           3.4.3
graphql-core                       3.2.5
graphql-relay                      3.2.0
greenlet                           3.1.1
grpcio                             1.69.0
gunicorn                           20.1.0
gviz-api                           1.10.0
h11                                0.14.0
h5py                               3.3.0
hijri-converter                    2.2.4
holidays                           0.15
horovod                            0.25.0
htmlmin                            0.1.12
httpcore                           1.0.7
httpx                              0.28.1
huggingface-hub                    0.27.1
humanfriendly                      10.0
idna                               3.2
imagecodecs                        2024.12.30
ImageHash                          4.3.0
imageio                            2.36.1
imbalanced-learn                   0.8.1
importlib_metadata                 8.5.0
importlib_resources                6.5.2
ipykernel                          6.12.1
ipython                            7.32.0
ipython-genutils                   0.2.0
ipywidgets                         7.7.0
isodate                            0.6.1
itk                                5.4.0
itk-core                           5.4.0
itk-filtering                      5.4.0
itk-io                             5.4.0
itk-numerics                       5.4.0
itk-registration                   5.4.0
itk-segmentation                   5.4.0
itsdangerous                       2.0.1
jedi                               0.18.0
Jinja2                             2.11.3
jmespath                           0.10.0
joblib                             1.0.1
joblibspark                        0.5.0
json-tricks                        3.17.3
jsonschema                         3.2.0
jupyter-client                     6.1.12
jupyter-core                       4.8.1
jupyterlab-pygments                0.1.2
jupyterlab-widgets                 1.0.0
keras                              2.9.0
Keras-Preprocessing                1.1.2
kiwisolver                         1.3.1
korean-lunar-calendar              0.3.1
langcodes                          3.3.0
lazy_loader                        0.4
libclang                           14.0.6
lightgbm                           3.3.2
lightning-utilities                0.11.9
llvmlite                           0.37.0
lmdb                               1.6.2
lpips                              0.1.4
LunarCalendar                      0.0.9
Mako                               1.2.0
Markdown                           3.3.6
MarkupSafe                         2.0.1
matplotlib                         3.9.4
matplotlib-inline                  0.1.2
missingno                          0.5.1
mistune                            0.8.4
mleap                              0.20.0
mlflow                             2.19.0
mlflow-databricks-artifacts        2.0.0
mlflow-skinny                      2.19.0
monai                              1.4.0
monailabel                         0.8.5
mpmath                             1.3.0
multidict                          6.1.0
multimethod                        1.9
murmurhash                         1.0.8
mypy-extensions                    0.4.3
nbclient                           0.5.3
nbconvert                          6.1.0
nbformat                           5.1.3
nest-asyncio                       1.5.1
networkx                           3.2.1
nibabel                            5.3.2
ninja                              1.11.1.3
nltk                               3.6.5
nni                                3.0
notebook                           6.4.5
numba                              0.54.1
numcodecs                          0.12.1
numpy                              1.26.4
numpymaxflow                       0.0.7
nvidia-ml-py                       12.560.30
oauthlib                           3.2.0
onnx                               1.17.0
onnxruntime                        1.19.2
opencv-python                      4.11.0.86
openslide-python                   1.4.1
opentelemetry-api                  1.29.0
opentelemetry-sdk                  1.29.0
opentelemetry-semantic-conventions 0.50b0
opt-einsum                         3.3.0
optuna                             4.1.0
orderedmultidict                   1.0.1
packaging                          21.0
pandas                             1.3.4
pandas-profiling                   3.1.0
pandocfilters                      1.4.3
paramiko                           2.9.2
parso                              0.8.2
passlib                            1.7.4
pathlib2                           2.3.7.post1
pathspec                           0.9.0
pathy                              0.6.2
patsy                              0.5.2
petastorm                          0.11.4
pexpect                            4.8.0
phik                               0.12.2
pickleshare                        0.7.5
pillow                             11.1.0
pip                                21.2.4
platformdirs                       2.5.2
plotly                             5.9.0
pmdarima                           1.8.5
preshed                            3.0.7
prettytable                        3.12.0
prompt-toolkit                     3.0.20
propcache                          0.2.1
prophet                            1.0.1
protobuf                           5.29.3
psutil                             5.8.0
psycopg2                           2.9.3
ptyprocess                         0.7.0
pyamg                              5.2.1
pyarrow                            7.0.0
pyarrow-hotfix                     0.5
pyasn1                             0.4.8
pyasn1-modules                     0.2.8
pybind11                           2.10.0
pycparser                          2.20
pydantic                           2.10.5
pydantic_core                      2.27.2
pydantic-settings                  2.7.1
pydicom                            2.4.4
pydicom-seg                        0.4.1
Pygments                           2.10.0
PyGObject                          3.36.0
PyJWT                              2.10.1
PyMeeus                            0.5.11
PyNaCl                             1.5.0
pynetdicom                         2.0.2
pynrrd                             1.1.1
pyodbc                             4.0.31
pyparsing                          3.0.4
pyrsistent                         0.18.0
PySocks                            1.7.1
pystan                             2.19.1.1
python-apt                         2.0.1+ubuntu0.20.4.1
python-dateutil                    2.8.2
python-dotenv                      1.0.1
python-editor                      1.0.4
python-multipart                   0.0.20
PythonWebHDFS                      0.2.3
pytorch-ignite                     0.4.11
pytz                               2021.3
PyWavelets                         1.1.1
PyYAML                             6.0.2
pyzmq                              22.2.1
referencing                        0.36.1
regex                              2021.8.3
requests                           2.32.3
requests-oauthlib                  1.3.1
requests-toolbelt                  1.0.0
requests-unixsocket                0.2.0
responses                          0.25.6
retrying                           1.3.4
rpds-py                            0.22.3
rsa                                4.9
s3fs                               2024.12.0
s3transfer                         0.5.2
safetensors                        0.5.2
schedule                           1.2.2
schema                             0.7.7
scikit-image                       0.22.0
scikit-learn                       0.24.2
scipy                              1.13.1
seaborn                            0.11.2
Send2Trash                         1.8.0
setuptools                         58.0.4
setuptools-git                     1.2
shap                               0.41.0
shapely                            2.0.6
SimpleITK                          2.4.1
simplejson                         3.17.6
six                                1.16.0
slicer                             0.0.7
smart-open                         5.2.1
smmap                              5.0.0
sniffio                            1.3.1
sortedcontainers                   2.4.0
soupsieve                          2.6
spacy                              3.4.1
spacy-legacy                       3.0.10
spacy-loggers                      1.0.3
spark-tensorflow-distributor       1.0.0
SQLAlchemy                         2.0.37
sqlparse                           0.4.2
srsly                              2.4.4
ssh-import-id                      5.10
starlette                          0.41.3
statsmodels                        0.12.2
sympy                              1.13.3
tabulate                           0.8.9
tangled-up-in-unicode              0.1.0
tenacity                           8.0.1
tensorboard                        2.18.0
tensorboard-data-server            0.7.2
tensorboard-plugin-profile         2.8.0
tensorboard-plugin-wit             1.8.1
tensorboardX                       2.6.2.2
tensorflow                         2.9.1
tensorflow-estimator               2.9.0
tensorflow-io-gcs-filesystem       0.27.0
termcolor                          2.0.1
terminado                          0.9.4
testpath                           0.5.0
thinc                              8.1.2
threadpoolctl                      2.2.0
tifffile                           2024.8.30
timeloop                           1.0.2
tokenize-rt                        4.2.1
tokenizers                         0.19.1
tomli                              2.0.1
torch                              1.12.1+cu113
torchmetrics                       1.5.2
torchvision                        0.13.1+cu113
tornado                            6.1
tqdm                               4.62.3
traitlets                          5.1.0
transformers                       4.40.2
typeguard                          4.1.2
typer                              0.4.2
typing_extensions                  4.12.2
ujson                              4.0.2
unattended-upgrades                0.1
urllib3                            1.26.20
uvicorn                            0.34.0
virtualenv                         20.8.0
visions                            0.7.4
wasabi                             0.10.1
watchdog                           6.0.0
wcwidth                            0.2.5
webencodings                       0.5.1
websocket-client                   1.3.1
websockets                         14.1
Werkzeug                           2.0.2
wheel                              0.37.0
widgetsnbextension                 3.6.0
wrapt                              1.12.1
xgboost                            1.6.2
yarl                               1.18.3
zarr                               2.18.2
zipp                               3.21.0

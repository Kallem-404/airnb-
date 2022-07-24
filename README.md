# airnb-
AirBNB Price Optimizer
Structured and time series data
In this model we optimize 'price' for host, the place we covered is Amsterdam.

%matplotlib inline
%reload_ext autoreload
%autoreload 2
!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl && pip3 install torchvision
!pip3 install fastai==0.7.0
!pip3 install torchtext==0.2.3
!pip3 install -U pandas
!pip3 install -U pandas_summary

!pip3 install requests
!pip3 install geopy

!mkdir airdata
%cd airdata/

!rm -rf *.csv
!rm -rf *.gz

!wget https://www.dropbox.com/s/apnxhwdkt5cqmkr/listings.tar.gz
!wget https://www.dropbox.com/s/4etjplku640e9y7/listings-test.csv  
!tar -zxvf listings.tar.gz
Collecting torch==0.3.0.post4 from http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
  Downloading http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl (592.3MB)
    100% |████████████████████████████████| 592.3MB 49.7MB/s 
Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from torch==0.3.0.post4) (3.13)
Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torch==0.3.0.post4) (1.14.6)
torchvision 0.2.1 has requirement pillow>=4.1.1, but you'll have pillow 4.0.0 which is incompatible.
fastai 1.0.46 has requirement numpy>=1.15, but you'll have numpy 1.14.6 which is incompatible.
fastai 1.0.46 has requirement torch>=1.0.0, but you'll have torch 0.3.0.post4 which is incompatible.
Installing collected packages: torch
  Found existing installation: torch 1.0.1.post2
    Uninstalling torch-1.0.1.post2:
      Successfully uninstalled torch-1.0.1.post2
Successfully installed torch-0.3.0.post4
Requirement already satisfied: torchvision in /usr/local/lib/python3.6/dist-packages (0.2.1)
Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torchvision) (1.14.6)
Requirement already satisfied: torch in /usr/local/lib/python3.6/dist-packages (from torchvision) (0.3.0.post4)
Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from torchvision) (1.11.0)
Collecting pillow>=4.1.1 (from torchvision)
  Downloading https://files.pythonhosted.org/packages/85/5e/e91792f198bbc5a0d7d3055ad552bc4062942d27eaf75c3e2783cf64eae5/Pillow-5.4.1-cp36-cp36m-manylinux1_x86_64.whl (2.0MB)
    100% |████████████████████████████████| 2.0MB 10.7MB/s 
Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from torch->torchvision) (3.13)
imgaug 0.2.8 has requirement numpy>=1.15.0, but you'll have numpy 1.14.6 which is incompatible.
fastai 1.0.46 has requirement numpy>=1.15, but you'll have numpy 1.14.6 which is incompatible.
fastai 1.0.46 has requirement torch>=1.0.0, but you'll have torch 0.3.0.post4 which is incompatible.
albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.8 which is incompatible.
Installing collected packages: pillow
  Found existing installation: Pillow 4.0.0
    Uninstalling Pillow-4.0.0:
      Successfully uninstalled Pillow-4.0.0
Successfully installed pillow-5.4.1
Collecting fastai==0.7.0
  Downloading https://files.pythonhosted.org/packages/50/6d/9d0d6e17a78b0598d5e8c49a0d03ffc7ff265ae62eca3e2345fab14edb9b/fastai-0.7.0-py3-none-any.whl (112kB)
    100% |████████████████████████████████| 122kB 4.2MB/s 
Collecting bcolz (from fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/5c/4e/23942de9d5c0fb16f10335fa83e52b431bcb8c0d4a8419c9ac206268c279/bcolz-1.2.1.tar.gz (1.5MB)
    100% |████████████████████████████████| 1.5MB 15.0MB/s 
Requirement already satisfied: decorator in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (4.3.2)
Requirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (2.5.3)
Requirement already satisfied: tornado in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (4.5.3)
Requirement already satisfied: ipykernel in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (4.6.1)
Requirement already satisfied: Pygments in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (2.1.3)
Collecting isoweek (from fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/c2/d4/fe7e2637975c476734fcbf53776e650a29680194eb0dd21dbdc020ca92de/isoweek-1.3.3-py2.py3-none-any.whl
Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (1.14.6)
Requirement already satisfied: opencv-python in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (3.4.5.20)
Requirement already satisfied: torchvision in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.2.1)
Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.2.0)
Requirement already satisfied: Jinja2 in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (2.10)
Requirement already satisfied: PyYAML in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (3.13)
Requirement already satisfied: torchtext in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.3.1)
Requirement already satisfied: pickleshare in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.7.5)
Collecting jedi (from fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/25/2b/1f188901be099d52d7b06f4d3b7cb9f8f09692c50697b139eaf6fa2928d8/jedi-0.13.3-py2.py3-none-any.whl (178kB)
    100% |████████████████████████████████| 184kB 11.2MB/s 
Requirement already satisfied: simplegeneric in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.8.1)
Collecting sklearn-pandas (from fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/1f/48/4e1461d828baf41d609efaa720d20090ac6ec346b5daad3c88e243e2207e/sklearn_pandas-1.8.0-py2.py3-none-any.whl
Requirement already satisfied: ipywidgets in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (7.4.2)
Requirement already satisfied: bleach in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (3.1.0)
Requirement already satisfied: pytz in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (2018.9)
Requirement already satisfied: ptyprocess in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.6.0)
Requirement already satisfied: widgetsnbextension in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (3.4.2)
Requirement already satisfied: torch<0.4 in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.3.0.post4)
Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (2018.11.29)
Requirement already satisfied: wcwidth in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.1.7)
Requirement already satisfied: graphviz in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.10.1)
Requirement already satisfied: Pillow in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (5.4.1)
Requirement already satisfied: entrypoints in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.3)
Requirement already satisfied: traitlets in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (4.3.2)
Requirement already satisfied: webencodings in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.5.1)
Requirement already satisfied: pyparsing in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (2.3.1)
Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (1.1.0)
Requirement already satisfied: jupyter in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (1.0.0)
Collecting pandas-summary (from fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/6b/00/f7b4d7fd901db9a79d63e88000bd1e12efba4f5fb52608f906d7fea2b18f/pandas_summary-0.0.6-py2.py3-none-any.whl
Collecting feather-format (from fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/08/55/940b97cc6f19a19f5dab9efef2f68a0ce43a7632f858b272391f0b851a7e/feather-format-0.4.0.tar.gz
Requirement already satisfied: jsonschema in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (2.6.0)
Requirement already satisfied: testpath in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.4.2)
Requirement already satisfied: pyzmq in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (17.0.0)
Requirement already satisfied: cycler in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.10.0)
Requirement already satisfied: MarkupSafe in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (1.1.1)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (3.0.2)
Collecting plotnine (from fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/84/02/b171c828560aea3a5da1efda464230dac3ef4f4834b88e0bd52ad14a08f0/plotnine-0.5.1-py2.py3-none-any.whl (3.6MB)
    100% |████████████████████████████████| 3.6MB 10.6MB/s 
Requirement already satisfied: seaborn in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.7.1)
Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (4.28.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (0.22.0)
Requirement already satisfied: ipython in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (5.5.0)
Requirement already satisfied: html5lib in /usr/local/lib/python3.6/dist-packages (from fastai==0.7.0) (1.0.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil->fastai==0.7.0) (1.11.0)
Requirement already satisfied: jupyter-client in /usr/local/lib/python3.6/dist-packages (from ipykernel->fastai==0.7.0) (5.2.4)
Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from torchtext->fastai==0.7.0) (2.18.4)
Collecting parso>=0.3.0 (from jedi->fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/19/b1/522b2671cc6d134c9d3f5dfc0d02fee07cab848e908d03d2bffea78cca8f/parso-0.3.4-py2.py3-none-any.whl (93kB)
    100% |████████████████████████████████| 102kB 29.7MB/s 
Requirement already satisfied: scikit-learn>=0.15.0 in /usr/local/lib/python3.6/dist-packages (from sklearn-pandas->fastai==0.7.0) (0.20.2)
Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets->fastai==0.7.0) (4.4.0)
Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.6/dist-packages (from widgetsnbextension->fastai==0.7.0) (5.2.2)
Requirement already satisfied: qtconsole in /usr/local/lib/python3.6/dist-packages (from jupyter->fastai==0.7.0) (4.4.3)
Requirement already satisfied: jupyter-console in /usr/local/lib/python3.6/dist-packages (from jupyter->fastai==0.7.0) (6.0.0)
Requirement already satisfied: nbconvert in /usr/local/lib/python3.6/dist-packages (from jupyter->fastai==0.7.0) (5.4.1)
Collecting pyarrow>=0.4.0 (from feather-format->fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/13/37/eb9aefcd6a041dffb4db6729daea2a91a01a1bf9815e02a3d35281348a85/pyarrow-0.12.1-cp36-cp36m-manylinux1_x86_64.whl (12.4MB)
    100% |████████████████████████████████| 12.4MB 3.9MB/s 
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->fastai==0.7.0) (1.0.1)
Requirement already satisfied: patsy>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from plotnine->fastai==0.7.0) (0.5.1)
Collecting mizani>=0.5.2 (from plotnine->fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/21/00/f9b9334c94fe0f5267c81697b3f8a85b360144b3840f08526344c4dcae72/mizani-0.5.3-py2.py3-none-any.whl (59kB)
    100% |████████████████████████████████| 61kB 22.2MB/s 
Requirement already satisfied: statsmodels>=0.8.0 in /usr/local/lib/python3.6/dist-packages (from plotnine->fastai==0.7.0) (0.8.0)
Collecting descartes>=1.1.0 (from plotnine->fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/e5/b6/1ed2eb03989ae574584664985367ba70cd9cf8b32ee8cad0e8aaeac819f3/descartes-1.1.0-py3-none-any.whl
Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.6/dist-packages (from ipython->fastai==0.7.0) (1.0.15)
Requirement already satisfied: pexpect; sys_platform != "win32" in /usr/local/lib/python3.6/dist-packages (from ipython->fastai==0.7.0) (4.6.0)
Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.6/dist-packages (from ipython->fastai==0.7.0) (40.8.0)
Requirement already satisfied: jupyter-core in /usr/local/lib/python3.6/dist-packages (from jupyter-client->ipykernel->fastai==0.7.0) (4.4.0)
Requirement already satisfied: urllib3<1.23,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext->fastai==0.7.0) (1.22)
Requirement already satisfied: idna<2.7,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext->fastai==0.7.0) (2.6)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext->fastai==0.7.0) (3.0.4)
Requirement already satisfied: terminado>=0.3.3; sys_platform != "win32" in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension->fastai==0.7.0) (0.8.1)
Requirement already satisfied: mistune>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->fastai==0.7.0) (0.8.4)
Requirement already satisfied: defusedxml in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->fastai==0.7.0) (0.5.0)
Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->jupyter->fastai==0.7.0) (1.4.2)
Collecting palettable (from mizani>=0.5.2->plotnine->fastai==0.7.0)
  Downloading https://files.pythonhosted.org/packages/56/8a/84537c0354f0d1f03bf644b71bf8e0a50db9c1294181905721a5f3efbf66/palettable-3.1.1-py2.py3-none-any.whl (77kB)
    100% |████████████████████████████████| 81kB 24.4MB/s 
Building wheels for collected packages: bcolz, feather-format
  Building wheel for bcolz (setup.py) ... done
  Stored in directory: /root/.cache/pip/wheels/9f/78/26/fb8c0acb91a100dc8914bf236c4eaa4b207cb876893c40b745
  Building wheel for feather-format (setup.py) ... done
  Stored in directory: /root/.cache/pip/wheels/85/7d/12/2dfa5c0195f921ac935f5e8f27deada74972edc0ae9988a9c1
Successfully built bcolz feather-format
mizani 0.5.3 has requirement pandas>=0.23.4, but you'll have pandas 0.22.0 which is incompatible.
plotnine 0.5.1 has requirement pandas>=0.23.4, but you'll have pandas 0.22.0 which is incompatible.
Installing collected packages: bcolz, isoweek, parso, jedi, sklearn-pandas, pandas-summary, pyarrow, feather-format, palettable, mizani, descartes, plotnine, fastai
  Found existing installation: fastai 1.0.46
    Uninstalling fastai-1.0.46:
      Successfully uninstalled fastai-1.0.46
Successfully installed bcolz-1.2.1 descartes-1.1.0 fastai-0.7.0 feather-format-0.4.0 isoweek-1.3.3 jedi-0.13.3 mizani-0.5.3 palettable-3.1.1 pandas-summary-0.0.6 parso-0.3.4 plotnine-0.5.1 pyarrow-0.12.1 sklearn-pandas-1.8.0
Collecting torchtext==0.2.3
  Downloading https://files.pythonhosted.org/packages/78/90/474d5944d43001a6e72b9aaed5c3e4f77516fbef2317002da2096fd8b5ea/torchtext-0.2.3.tar.gz (42kB)
    100% |████████████████████████████████| 51kB 2.5MB/s 
Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from torchtext==0.2.3) (4.28.1)
Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from torchtext==0.2.3) (2.18.4)
Requirement already satisfied: urllib3<1.23,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext==0.2.3) (1.22)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext==0.2.3) (3.0.4)
Requirement already satisfied: idna<2.7,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext==0.2.3) (2.6)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext==0.2.3) (2018.11.29)
Building wheels for collected packages: torchtext
  Building wheel for torchtext (setup.py) ... done
  Stored in directory: /root/.cache/pip/wheels/42/a6/f4/b267328bde6bb680094a0c173e8e5627ccc99543abded97204
Successfully built torchtext
Installing collected packages: torchtext
  Found existing installation: torchtext 0.3.1
    Uninstalling torchtext-0.3.1:
      Successfully uninstalled torchtext-0.3.1
Successfully installed torchtext-0.2.3
Collecting pandas
  Downloading https://files.pythonhosted.org/packages/e6/de/a0d3defd8f338eaf53ef716e40ef6d6c277c35d50e09b586e170169cdf0d/pandas-0.24.1-cp36-cp36m-manylinux1_x86_64.whl (10.1MB)
    100% |████████████████████████████████| 10.1MB 4.3MB/s 
Requirement already satisfied, skipping upgrade: python-dateutil>=2.5.0 in /usr/local/lib/python3.6/dist-packages (from pandas) (2.5.3)
Requirement already satisfied, skipping upgrade: numpy>=1.12.0 in /usr/local/lib/python3.6/dist-packages (from pandas) (1.14.6)
Requirement already satisfied, skipping upgrade: pytz>=2011k in /usr/local/lib/python3.6/dist-packages (from pandas) (2018.9)
Requirement already satisfied, skipping upgrade: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.5.0->pandas) (1.11.0)
pymc3 3.6 has requirement joblib<0.13.0, but you'll have joblib 0.13.2 which is incompatible.
Installing collected packages: pandas
  Found existing installation: pandas 0.22.0
    Uninstalling pandas-0.22.0:
      Successfully uninstalled pandas-0.22.0
Successfully installed pandas-0.24.1
Requirement already up-to-date: pandas_summary in /usr/local/lib/python3.6/dist-packages (0.0.6)
Requirement already satisfied, skipping upgrade: pandas in /usr/local/lib/python3.6/dist-packages (from pandas_summary) (0.24.1)
Requirement already satisfied, skipping upgrade: numpy in /usr/local/lib/python3.6/dist-packages (from pandas_summary) (1.14.6)
Requirement already satisfied, skipping upgrade: python-dateutil>=2.5.0 in /usr/local/lib/python3.6/dist-packages (from pandas->pandas_summary) (2.5.3)
Requirement already satisfied, skipping upgrade: pytz>=2011k in /usr/local/lib/python3.6/dist-packages (from pandas->pandas_summary) (2018.9)
Requirement already satisfied, skipping upgrade: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.5.0->pandas->pandas_summary) (1.11.0)
Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (2.18.4)
Requirement already satisfied: urllib3<1.23,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests) (1.22)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests) (2018.11.29)
Requirement already satisfied: idna<2.7,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests) (2.6)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests) (3.0.4)
Requirement already satisfied: geopy in /usr/local/lib/python3.6/dist-packages (1.17.0)
Requirement already satisfied: geographiclib<2,>=1.49 in /usr/local/lib/python3.6/dist-packages (from geopy) (1.49)
/content/airdata
--2019-03-01 23:28:45--  https://www.dropbox.com/s/apnxhwdkt5cqmkr/listings.tar.gz
Resolving www.dropbox.com (www.dropbox.com)... 162.125.8.1, 2620:100:6018:1::a27d:301
Connecting to www.dropbox.com (www.dropbox.com)|162.125.8.1|:443... connected.
HTTP request sent, awaiting response... 301 Moved Permanently
Location: /s/raw/apnxhwdkt5cqmkr/listings.tar.gz [following]
--2019-03-01 23:28:45--  https://www.dropbox.com/s/raw/apnxhwdkt5cqmkr/listings.tar.gz
Reusing existing connection to www.dropbox.com:443.
HTTP request sent, awaiting response... 302 Found
Location: https://uc9392c7fc56d739fad087f96489.dl.dropboxusercontent.com/cd/0/inline/AcRgV4jJKQuxcOG4MBZvdZZaUZwdrQSVwWDDk0m9mikc6K7AoLm7ULM3UIvfoRlPSRZ8UglemSdE-GYz9Tndsue-HgRZ4GXgXn9gLTsL6tZlYw/file# [following]
--2019-03-01 23:28:45--  https://uc9392c7fc56d739fad087f96489.dl.dropboxusercontent.com/cd/0/inline/AcRgV4jJKQuxcOG4MBZvdZZaUZwdrQSVwWDDk0m9mikc6K7AoLm7ULM3UIvfoRlPSRZ8UglemSdE-GYz9Tndsue-HgRZ4GXgXn9gLTsL6tZlYw/file
Resolving uc9392c7fc56d739fad087f96489.dl.dropboxusercontent.com (uc9392c7fc56d739fad087f96489.dl.dropboxusercontent.com)... 162.125.3.6, 2620:100:601b:6::a27d:806
Connecting to uc9392c7fc56d739fad087f96489.dl.dropboxusercontent.com (uc9392c7fc56d739fad087f96489.dl.dropboxusercontent.com)|162.125.3.6|:443... connected.
HTTP request sent, awaiting response... 302 FOUND
Location: /cd/0/inline2/AcSgk0ZvhqeEZaqWCTbQh5-mtrb8_lpWbzx1-3GhVKygnrPuhb3IOxt4djfaYamDaEsVxX8FBsUT8qCICRu0PhcIHMdj0HbOpCL4FTUhXnMz7-cr3fu-MFqDRedCbVoujuvkSo6Am8XDPN3PiairFcCfV45V4YXzH90C2yrkoj0TZYdQ_Y6Bmds_W0Jp9rjP68bh-Vdn-Ys9JMcU0UXZo7BXDiT7KS4rRVoaqMiKxGhGEExalUBtlnpLaGDl739p_U3pMuqC61a7Mr4HlonDqsqUVR-UewSpcYHhL3_mAbfZuvasZbFZYxHdRxPsKgwyas3Ro4YXqYI4Jq0xfHEHd8n-/file [following]
--2019-03-01 23:28:46--  https://uc9392c7fc56d739fad087f96489.dl.dropboxusercontent.com/cd/0/inline2/AcSgk0ZvhqeEZaqWCTbQh5-mtrb8_lpWbzx1-3GhVKygnrPuhb3IOxt4djfaYamDaEsVxX8FBsUT8qCICRu0PhcIHMdj0HbOpCL4FTUhXnMz7-cr3fu-MFqDRedCbVoujuvkSo6Am8XDPN3PiairFcCfV45V4YXzH90C2yrkoj0TZYdQ_Y6Bmds_W0Jp9rjP68bh-Vdn-Ys9JMcU0UXZo7BXDiT7KS4rRVoaqMiKxGhGEExalUBtlnpLaGDl739p_U3pMuqC61a7Mr4HlonDqsqUVR-UewSpcYHhL3_mAbfZuvasZbFZYxHdRxPsKgwyas3Ro4YXqYI4Jq0xfHEHd8n-/file
Reusing existing connection to uc9392c7fc56d739fad087f96489.dl.dropboxusercontent.com:443.
HTTP request sent, awaiting response... 200 OK
Length: 199411704 (190M) [application/octet-stream]
Saving to: ‘listings.tar.gz’

listings.tar.gz     100%[===================>] 190.17M  53.4MB/s    in 3.6s    

2019-03-01 23:28:50 (53.4 MB/s) - ‘listings.tar.gz’ saved [199411704/199411704]

--2019-03-01 23:28:51--  https://www.dropbox.com/s/4etjplku640e9y7/listings-test.csv
Resolving www.dropbox.com (www.dropbox.com)... 162.125.8.1, 2620:100:6018:1::a27d:301
Connecting to www.dropbox.com (www.dropbox.com)|162.125.8.1|:443... connected.
HTTP request sent, awaiting response... 301 Moved Permanently
Location: /s/raw/4etjplku640e9y7/listings-test.csv [following]
--2019-03-01 23:28:51--  https://www.dropbox.com/s/raw/4etjplku640e9y7/listings-test.csv
Reusing existing connection to www.dropbox.com:443.
HTTP request sent, awaiting response... 302 Found
Location: https://uc99b5116f01f5b9a0e0db347eee.dl.dropboxusercontent.com/cd/0/inline/AcSqxlOsJ9R2gHje70gE2h6gPWNwx_xMCjfCukhROWof8Kne98eOpteL0UCFOKl-Q3v7p2mPjlLlbbQf_iC9T9oj_NelVzFfwABV_VqI6Gc5jg/file# [following]
--2019-03-01 23:28:52--  https://uc99b5116f01f5b9a0e0db347eee.dl.dropboxusercontent.com/cd/0/inline/AcSqxlOsJ9R2gHje70gE2h6gPWNwx_xMCjfCukhROWof8Kne98eOpteL0UCFOKl-Q3v7p2mPjlLlbbQf_iC9T9oj_NelVzFfwABV_VqI6Gc5jg/file
Resolving uc99b5116f01f5b9a0e0db347eee.dl.dropboxusercontent.com (uc99b5116f01f5b9a0e0db347eee.dl.dropboxusercontent.com)... 162.125.8.6, 2620:100:601b:6::a27d:806
Connecting to uc99b5116f01f5b9a0e0db347eee.dl.dropboxusercontent.com (uc99b5116f01f5b9a0e0db347eee.dl.dropboxusercontent.com)|162.125.8.6|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 418871 (409K) [text/plain]
Saving to: ‘listings-test.csv’

listings-test.csv   100%[===================>] 409.05K  --.-KB/s    in 0.1s    

2019-03-01 23:28:52 (3.90 MB/s) - ‘listings-test.csv’ saved [418871/418871]

./._amsterdam
./amsterdam/
./amsterdam/._listings.csv-oct17
./amsterdam/listings.csv-oct17
./amsterdam/._listings.csv-nov17
./amsterdam/listings.csv-nov17
./amsterdam/._listings.csv-may
./amsterdam/listings.csv-may
./amsterdam/._listings.csv-july
./amsterdam/listings.csv-july
./amsterdam/._listings.csv-jan17
./amsterdam/listings.csv-jan17
./amsterdam/._listings.csv-aug17
./amsterdam/listings.csv-aug17
./amsterdam/._listings.csv-april17
./amsterdam/listings.csv-april17
./amsterdam/._listings.csv-may17
./amsterdam/listings.csv-may17
./amsterdam/._listings.csv-april
./amsterdam/listings.csv-april
./amsterdam/._listings.csv-aug
./amsterdam/listings.csv-aug
./amsterdam/._listings.csv-july17
./amsterdam/listings.csv-july17
./amsterdam/._listings.csv-june
./amsterdam/listings.csv-june
!rm ./amsterdam/._listings*
!ls -la ./amsterdam/
rm: cannot remove './amsterdam/._listings*': No such file or directory
total 827412
drwxr-xr-x 2  501 staff     4096 Mar  1 23:29 .
drwxr-xr-x 3 root root      4096 Mar  1 23:28 ..
-rw-r--r-- 1  501 staff 72373058 Aug 29  2018 listings.csv-april
-rw-r--r-- 1  501 staff 61029549 Aug 29  2018 listings.csv-april17
-rw-r--r-- 1  501 staff 78346953 Aug 29  2018 listings.csv-aug
-rw-r--r-- 1  501 staff 74750471 Aug 29  2018 listings.csv-aug17
-rw-r--r-- 1  501 staff 61944179 Aug 29  2018 listings.csv-jan17
-rw-r--r-- 1  501 staff 75613566 Aug 29  2018 listings.csv-july
-rw-r--r-- 1  501 staff 67530539 Aug 29  2018 listings.csv-july17
-rw-r--r-- 1  501 staff 74340672 Aug 29  2018 listings.csv-june
-rw-r--r-- 1  501 staff 72223308 Aug 29  2018 listings.csv-may
-rw-r--r-- 1  501 staff 62795363 Aug 29  2018 listings.csv-may17
-rw-r--r-- 1  501 staff 71996844 Aug 29  2018 listings.csv-nov17
-rw-r--r-- 1  501 staff 74294362 Aug 29  2018 listings.csv-oct17
import json
import pandas as pd
import requests
import csv


from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
np.set_printoptions(threshold=50, edgeitems=20)

PATH='amsterdam/'

from IPython.display import HTML, display
Set pandas display options for custom rendering.

#pd.set_option('display.height', 700)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

set_plot_sizes(12,14,16)
directory = PATH
table_names = []

for filename in os.listdir(directory):
  if filename.startswith("listings.csv"):
    print(os.path.join(directory, filename))
    table_names.append(filename)
    continue
  else:
    print("ERROR:", os.path.join(directory, filename))
    continue
amsterdam/listings.csv-may17
amsterdam/listings.csv-april17
amsterdam/listings.csv-nov17
amsterdam/listings.csv-april
amsterdam/listings.csv-july17
amsterdam/listings.csv-june
amsterdam/listings.csv-july
amsterdam/listings.csv-aug17
amsterdam/listings.csv-oct17
amsterdam/listings.csv-aug
amsterdam/listings.csv-jan17
amsterdam/listings.csv-may
tables = [pd.read_csv(f'{PATH}{fname}', low_memory=False) for fname in table_names]
for t in tables: display(t.head())
listing_df = pd.concat([t for t in tables])
listing_df.shape
(214306, 96)
Test Data
'price' is not present in the test data.

Data Cleaning / Feature Engineering
As a structured data problem, we necessarily have to go through all the cleaning and feature engineering, even though we're using a neural network.

test_df = pd.read_csv('listings-test.csv', low_memory=False)
test_df['num_host_verifications'] = 0
# Drop following colums from the listing.csv
test_df.drop(['listing_url','scrape_id','name','summary','space','description','experiences_offered','neighborhood_overview','notes','transit',
                 'access','interaction','house_rules','thumbnail_url','medium_url','picture_url','xl_picture_url','host_id','host_url','host_name',
                 'host_since','host_location','host_about','host_response_time','host_response_rate','host_acceptance_rate','host_thumbnail_url',
                 'host_picture_url','host_neighbourhood','host_listings_count','host_has_profile_pic','neighbourhood','neighbourhood_cleansed',
                 'neighbourhood_group_cleansed','street','zipcode','market','country_code','is_location_exact','square_feet','calendar_updated',
                 'calendar_last_scraped','first_review','last_review','requires_license','license','jurisdiction_names','cancellation_policy',
                 'require_guest_profile_picture','require_guest_phone_verification','reviews_per_month','id'
                ], axis=1, inplace=True)


# Column name on calendar and listing table has to be same, for merge to happen.

#test_df.rename(columns={'id': 'listing_id'}, inplace=True)
test_df.rename(columns={'last_scraped': 'date'}, inplace=True)


add_datepart(test_df,'date',drop=False)

# Create seperate columns for amenities.

test_df['essentials'] = 0
test_df['kitchen'] = 0
test_df['air_conditioning'] = 0
test_df['washer'] = 0
test_df['tv'] = 0
test_df['wifi'] = 0
test_df['free_parking_on_premises'] = 0
test_df['gym'] = 0
test_df['pool'] = 0
test_df['breakfast'] = 0
test_df['num_host_verifications'] = 0


test_df['essentials'] = test_df.amenities.str.contains('Essentials')
test_df['kitchen'] = test_df.amenities.str.contains('Kitchen')
test_df['air_conditioning'] = test_df.amenities.str.contains('Air conditioning')
test_df['washer'] = test_df.amenities.str.contains('Washer')
test_df['tv'] = test_df.amenities.str.contains('TV')
test_df['wifi'] = test_df.amenities.str.contains('Wifi')
test_df['free_parking_on_premises'] = test_df.amenities.str.contains('Free parking on premises')
test_df['gym'] = test_df.amenities.str.contains('Gym')
test_df['pool'] = test_df.amenities.str.contains('Pool')
test_df['breakfast'] = test_df.amenities.str.contains('Breakfast')


test_df['num_host_verifications'] += test_df.host_verifications.str.contains('email')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('phone')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('manual_online')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('reviews')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('manual_offline')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('jumio')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('offline_government_id')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('selfie')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('government_id')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('identity_manual')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('work_email')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('facebook')
test_df['num_host_verifications'] += test_df.host_verifications.str.contains('google')


test_df.host_identity_verified = test_df.host_identity_verified!='f'
test_df.host_is_superhost = test_df.host_is_superhost!='f'
test_df.instant_bookable = test_df.instant_bookable!='f'
test_df.is_business_travel_ready = test_df.is_business_travel_ready!='f'
test_df.has_availability = test_df.has_availability!='f'

test_df['weekly_price'] = test_df['weekly_price'].astype(str)
test_df['weekly_price'] = test_df.weekly_price.apply(lambda x: x.replace('$',''))
test_df['weekly_price'] = test_df.weekly_price.apply(lambda x: x.replace(',',''))
test_df['weekly_price'] = test_df['weekly_price'].astype(str)

test_df['monthly_price'] = test_df['monthly_price'].astype(str)
test_df['monthly_price'] = test_df.monthly_price.apply(lambda x: x.replace('$',''))
test_df['monthly_price'] = test_df.monthly_price.apply(lambda x: x.replace(',',''))
test_df['monthly_price'] = test_df['monthly_price'].astype(str)

test_df['cleaning_fee'] = test_df['cleaning_fee'].astype(str)
test_df['cleaning_fee'] = test_df.cleaning_fee.apply(lambda x: x.replace('$',''))
test_df['cleaning_fee'] = test_df.cleaning_fee.apply(lambda x: x.replace(',',''))
test_df['cleaning_fee'] = test_df['cleaning_fee'].astype(str)

test_df['extra_people'] = test_df['extra_people'].astype(str)
test_df['extra_people'] = test_df.extra_people.apply(lambda x: x.replace('$',''))
test_df['extra_people'] = test_df.extra_people.apply(lambda x: x.replace(',',''))
test_df['extra_people'] = test_df['extra_people'].astype(str)

test_df['security_deposit'] = test_df['security_deposit'].astype(str)
test_df['security_deposit'] = test_df.security_deposit.apply(lambda x: x.replace('$',''))
test_df['security_deposit'] = test_df.security_deposit.apply(lambda x: x.replace(',',''))
test_df['security_deposit'] = test_df['security_deposit'].astype(str)

test_df['price'] = test_df['price'].astype(str)
test_df['price'] = test_df.price.apply(lambda x: x.replace('$',''))
test_df['price'] = test_df.price.apply(lambda x: x.replace(',',''))
test_df['price'] = test_df['price'].astype(float)
Data Cleaning / Feature Engineering
As a structured data problem, we necessarily have to go through all the cleaning and feature engineering, even though we're using a neural network.

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
geolocator = Nominatim(user_agent="Optima")
# Drop following colums from the listing.csv

listing_df.drop(['listing_url','scrape_id','name','summary','space','description','experiences_offered','neighborhood_overview','notes','transit',
                 'access','interaction','house_rules','thumbnail_url','medium_url','picture_url','xl_picture_url','host_id','host_url','host_name',
                 'host_since','host_location','host_about','host_response_time','host_response_rate','host_acceptance_rate','host_thumbnail_url',
                 'host_picture_url','host_neighbourhood','host_listings_count','host_has_profile_pic','neighbourhood','neighbourhood_cleansed',
                 'neighbourhood_group_cleansed','street','zipcode','market','country_code','is_location_exact','square_feet','calendar_updated',
                 'calendar_last_scraped','first_review','last_review','requires_license','license','jurisdiction_names','cancellation_policy',
                 'require_guest_profile_picture','require_guest_phone_verification','reviews_per_month'
                ], axis=1, inplace=True)


# Column name on calendar and listing table has to be same, for merge to happen.

listing_df.rename(columns={'id': 'listing_id'}, inplace=True)
listing_df.rename(columns={'last_scraped': 'date'}, inplace=True)


# Create seperate columns for amenities.

listing_df['essentials'] = 0
listing_df['kitchen'] = 0
listing_df['air_conditioning'] = 0
listing_df['washer'] = 0
listing_df['tv'] = 0
listing_df['wifi'] = 0
listing_df['free_parking_on_premises'] = 0
listing_df['gym'] = 0
listing_df['pool'] = 0
listing_df['breakfast'] = 0
listing_df['num_host_verifications'] = 0


listing_df['essentials'] = listing_df.amenities.str.contains('Essentials')
listing_df['kitchen'] = listing_df.amenities.str.contains('Kitchen')
listing_df['air_conditioning'] = listing_df.amenities.str.contains('Air conditioning')
listing_df['washer'] = listing_df.amenities.str.contains('Washer')
listing_df['tv'] = listing_df.amenities.str.contains('TV')
listing_df['wifi'] = listing_df.amenities.str.contains('Wifi')
listing_df['free_parking_on_premises'] = listing_df.amenities.str.contains('Free parking on premises')
listing_df['gym'] = listing_df.amenities.str.contains('Gym')
listing_df['pool'] = listing_df.amenities.str.contains('Pool')
listing_df['breakfast'] = listing_df.amenities.str.contains('Breakfast')


listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('email')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('phone')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('manual_online')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('reviews')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('manual_offline')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('jumio')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('offline_government_id')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('selfie')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('government_id')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('identity_manual')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('work_email')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('facebook')
listing_df['num_host_verifications'] += listing_df.host_verifications.str.contains('google')

#list(listing_df)
Make following boolean that system can understand:

listing_df.host_identity_verified = listing_df.host_identity_verified!='f'
listing_df.host_is_superhost = listing_df.host_is_superhost!='f'
listing_df.instant_bookable = listing_df.instant_bookable!='f'
listing_df.is_business_travel_ready = listing_df.is_business_travel_ready!='f'
listing_df.has_availability = listing_df.has_availability!='f'
Make date more verbose:

# Break data into: year, month, week, day, day_of_week, day_of_month, ...
add_datepart(listing_df,'date',drop=False)
Remove special charaters from data like: $, comma, etc.

listing_df['weekly_price'] = listing_df['weekly_price'].astype(str)
listing_df['weekly_price'] = listing_df.weekly_price.apply(lambda x: x.replace('$',''))
listing_df['weekly_price'] = listing_df.weekly_price.apply(lambda x: x.replace(',',''))
listing_df['weekly_price'] = listing_df['weekly_price'].astype(str)

listing_df['monthly_price'] = listing_df['monthly_price'].astype(str)
listing_df['monthly_price'] = listing_df.monthly_price.apply(lambda x: x.replace('$',''))
listing_df['monthly_price'] = listing_df.monthly_price.apply(lambda x: x.replace(',',''))
listing_df['monthly_price'] = listing_df['monthly_price'].astype(str)

listing_df['cleaning_fee'] = listing_df['cleaning_fee'].astype(str)
listing_df['cleaning_fee'] = listing_df.cleaning_fee.apply(lambda x: x.replace('$',''))
listing_df['cleaning_fee'] = listing_df.cleaning_fee.apply(lambda x: x.replace(',',''))
listing_df['cleaning_fee'] = listing_df['cleaning_fee'].astype(str)

listing_df['extra_people'] = listing_df['extra_people'].astype(str)
listing_df['extra_people'] = listing_df.extra_people.apply(lambda x: x.replace('$',''))
listing_df['extra_people'] = listing_df.extra_people.apply(lambda x: x.replace(',',''))
listing_df['extra_people'] = listing_df['extra_people'].astype(str)

listing_df['security_deposit'] = listing_df['security_deposit'].astype(str)
listing_df['security_deposit'] = listing_df.security_deposit.apply(lambda x: x.replace('$',''))
listing_df['security_deposit'] = listing_df.security_deposit.apply(lambda x: x.replace(',',''))
listing_df['security_deposit'] = listing_df['security_deposit'].astype(str)

listing_df['price'] = listing_df['price'].astype(str)
listing_df['price'] = listing_df.price.apply(lambda x: x.replace('$',''))
listing_df['price'] = listing_df.price.apply(lambda x: x.replace(',',''))
listing_df['price'] = listing_df['price'].astype(float)
listing_df.head().T.head(70)
Temporary datadframe to populate location based amenities

#Option 1 : Calculate amenities from scratch.
#Option 1

col_names =  ['smart_location', 'latitude', 'longitude']
tmp_df  = pd.DataFrame(columns = col_names)
Iterate through all the places in the rows and find unique places lat,lng. These are places lat,lng which is unsually the center of the place.

for index, row in listing_df.iterrows():
  
  address = row['smart_location']

  if not tmp_df[tmp_df.smart_location == address].empty:
    #print('FOUND')
    continue
  
  else:
    location = geolocator.geocode(address, timeout=60)
    if location:
      #print((location.latitude, location.longitude))
      tmp_df.loc[len(tmp_df)] = [address, location.latitude, location.longitude]
    else:
      tmp_df.loc[len(tmp_df)] = [address, row['latitude'], row['longitude']]
      #print((row['latitude'], row['longitude']))
    
    
  
print('Done')    
Done
Create Location based amenities

### Desired fields ###
#'airport'
#'restaurant'
#'amusement_park','zoo','museum','hindu_temple','church','city_hall','park','stadium','art_gallery'
#'bar','night_club'
#'bus_station','subway_station','train_station','taxi_stand'
#'car_rental'
#'clothing_store','convenience_store','department_store','grocery_or_supermarket','shopping_mall','store'
#'gym','spa'

### Active fields ###
#'airport'
#'restaurant'
#'bar'
#'bus_station','train_station'
#'shopping_mall'
# FIXME: Add following for US/Europe - amusement_park, museum, stadium, zoo, art_gallery, night_club

tmp_df['airport'] = 0
tmp_df['restaurant'] = 0
tmp_df['bar'] = 0
tmp_df['bus_station'] = 0
tmp_df['train_station'] = 0
tmp_df['shopping_mall'] = 0
# OPTIONAL : Incase you would like to download the geo_cord_smart_location table.
tmp_df.to_csv('geo_cord_smart_location.csv', index=False)
from google.colab import files
files.download('geo_cord_smart_location.csv')
### APIs ###
## Send Request ##
def send_req(lat_lng, type):
  radius='2000'

  req=requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json?location='+lat_lng+'&radius='+radius+'&type='+type+'&key=AIzaSyCqzqa3Vgo0Uz5cOsXGGff5f7vxipu4Ug0')
  #print(req)
  data = req.json()
  #print (data)
  return data


for index, row in tmp_df.iterrows():
  
  lat = row['latitude']
  lng = row['longitude']
  lat_lng = str(lat)+','+str(lng)
  #print(lat_lng)
  
  
  ## 1. Airport ##
  ## Count
  data = send_req(lat_lng, 'airport')
  num_airports = 0
  for result in data['results']:
    if 'types' not in result:
      continue
    else:
      for airport in result['types']:
        if airport == 'airport': 
          num_airports += 1
  
  tmp_df.at[index, 'airport'] = num_airports  
  
  
  ## 2. Restaurant ##
  ## Count rating
  data = send_req(lat_lng, 'restaurant')
  num_restaurants = 0
  for result in data['results']:
    if 'rating' not in result:
      continue
    else:
      if result[u'rating'] >= 3.5:
        num_restaurants += 1
   
  tmp_df.at[index, 'restaurant'] = num_restaurants
  
  
  ## 3. Bar ##
  ## Count rating
  data = send_req(lat_lng, 'bar')
  num_bar = 0
  for result in data['results']:
    if 'rating' not in result:
      continue
    else:
      if result[u'rating'] >= 3.5:
        num_bar += 1
        
  tmp_df.at[index, 'bar'] = num_bar
  
  
  ## 4. bus_station ##
  ## Count
  data = send_req(lat_lng, 'bus_station')
  num_bus_stations = 0
  for result in data['results']:
    if 'types' not in result:
      continue
    else:
      for airport in result['types']:
        if airport == 'bus_station': 
          num_bus_stations += 1
          
  tmp_df.at[index, 'bus_station'] = num_bus_stations
                
    

  ## 5. train_station ##
  ## Count
  data = send_req(lat_lng, 'train_station')
  num_train_stations = 0
  for result in data['results']:
    if 'types' not in result:
      continue
    else:
      for train_station in result['types']:
        if train_station == 'train_station': 
          num_train_stations += 1

  tmp_df.at[index, 'train_station'] = num_train_stations
                


  ## 6. shopping_mall ##
  ## Count rating
  data = send_req(lat_lng, 'shopping_mall')
  num_shopping_malls = 0
  for result in data['results']:
    if 'rating' not in result:
      continue
    else:
      if result[u'rating'] >= 3.5:
        num_shopping_malls += 1
    
  tmp_df.at[index, 'shopping_mall'] = num_shopping_malls
  
print('Done Google Places')
Done Google Places
# DOWNLOAD : precalculated DB : Incase you would like to download the smart_location table.

tmp_df.to_csv('smart_location.csv', index=False)
from google.colab import files
files.download('smart_location.csv')
#Option 2: If precalculated places are available fetch them to avoid computational time.
#Option 2

!wget https://www.dropbox.com/s/8mjv8ul0a81vykv/smart_location.csv
tmp_df = pd.read_csv('smart_location.csv', low_memory=False)
--2019-03-01 23:51:56--  https://www.dropbox.com/s/8mjv8ul0a81vykv/smart_location.csv
Resolving www.dropbox.com (www.dropbox.com)... 162.125.8.1, 2620:100:601b:1::a27d:801
Connecting to www.dropbox.com (www.dropbox.com)|162.125.8.1|:443... connected.
HTTP request sent, awaiting response... 301 Moved Permanently
Location: /s/raw/8mjv8ul0a81vykv/smart_location.csv [following]
--2019-03-01 23:51:56--  https://www.dropbox.com/s/raw/8mjv8ul0a81vykv/smart_location.csv
Reusing existing connection to www.dropbox.com:443.
HTTP request sent, awaiting response... 302 Found
Location: https://ucd2227c7465401bcfa68db93202.dl.dropboxusercontent.com/cd/0/inline/AcSDQUrd6n16yDL7rX9biFlBFjplRIBvnF1RPY_63HknVmNdFEwS0H7dLbxphw-T5z1MGMAgPDlACRkLEGEGQwCAvyEzu0Elop8y8mcRRuNM7w/file# [following]
--2019-03-01 23:51:56--  https://ucd2227c7465401bcfa68db93202.dl.dropboxusercontent.com/cd/0/inline/AcSDQUrd6n16yDL7rX9biFlBFjplRIBvnF1RPY_63HknVmNdFEwS0H7dLbxphw-T5z1MGMAgPDlACRkLEGEGQwCAvyEzu0Elop8y8mcRRuNM7w/file
Resolving ucd2227c7465401bcfa68db93202.dl.dropboxusercontent.com (ucd2227c7465401bcfa68db93202.dl.dropboxusercontent.com)... 162.125.8.6, 2620:100:6018:6::a27d:306
Connecting to ucd2227c7465401bcfa68db93202.dl.dropboxusercontent.com (ucd2227c7465401bcfa68db93202.dl.dropboxusercontent.com)|162.125.8.6|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4364 (4.3K) [text/plain]
Saving to: ‘smart_location.csv’

smart_location.csv  100%[===================>]   4.26K  --.-KB/s    in 0s      

2019-03-01 23:51:56 (337 MB/s) - ‘smart_location.csv’ saved [4364/4364]

tmp_df
df_copy = tmp_df
df_copy.drop(['latitude','longitude'], axis=1, inplace=True)
df_copy
joined_df = pd.merge(listing_df,df_copy,on='smart_location')
# Time to drop few columns from dataframe.
# amenities, smart_location, square_feet, calendar_updated, cancellation_policy, ...
joined_df.drop(['amenities'], axis=1, inplace=True)
joined_df.drop(['smart_location'], axis=1, inplace=True)
joined_df.drop(['host_verifications'], axis=1, inplace=True)
joined_df.head().T.head(70)
display(DataFrameSummary(joined_df).summary())
For test data

joined_testdf = pd.merge(test_df,df_copy,on='smart_location')
# Time to drop few columns from dataframe.
# amenities, smart_location, square_feet, calendar_updated, cancellation_policy, ...
joined_testdf.drop(['amenities'], axis=1, inplace=True)
joined_testdf.drop(['smart_location'], axis=1, inplace=True)
joined_testdf.drop(['host_verifications'], axis=1, inplace=True)
display(DataFrameSummary(joined_testdf).summary())
Seperate out the train and test data based on the price availability.

Create Features
# FIXME: Include listing_id or NOT
cat_vars = ['price','bed_type','host_is_superhost','city','state','property_type','room_type','beds','bedrooms','bathrooms','instant_bookable',
            'host_identity_verified','host_total_listings_count','calculated_host_listings_count','is_business_travel_ready','monthly_price',
            'minimum_nights','maximum_nights','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
            'review_scores_communication','review_scores_location','review_scores_value','security_deposit','cleaning_fee','accommodates','Year','Month',
            'Week','Day','Dayofweek']



contin_vars = ['availability_30','availability_365','availability_60','availability_90','guests_included','number_of_reviews','latitude','longitude',
               'extra_people']


n = len(joined_df); n
214306
joined_testdf.head()
dep = 'listing_id'
joined_traindf = joined_df[cat_vars+contin_vars+[dep, 'date']].copy()
joined_testdf[dep] = 0
joined_testdf = joined_testdf[cat_vars+contin_vars+[dep, 'date']].copy()
Set the type of 'cat' variables as 'category'

for v in cat_vars: joined_traindf[v] = joined_traindf[v].astype('category').cat.as_ordered()
for v in cat_vars: joined_testdf[v] = joined_testdf[v].astype('category').cat.as_ordered()
apply_cats(joined_testdf, joined_traindf)
for v in contin_vars:
    joined_traindf[v] = joined_traindf[v].fillna(0).astype('float32')
    joined_testdf[v] = joined_testdf[v].fillna(0).astype('float32')
joined_traindf = joined_traindf[joined_traindf.price != 0]
Run of sample set or Full set depends upon the listing-small.csv or listing.csv. We are going to run on sample:

idxs = get_cv_idxs(n, val_pct=150000/n)
joined_samp = joined_traindf.iloc[idxs].set_index("date")
samp_size = len(joined_samp); samp_size
Run on full set, use this instead.

samp_size = n
joined_samp = joined_traindf.set_index("date")
display(DataFrameSummary(joined_samp).summary())
We can now process our data ...

df, y, nas, mapper = proc_df(joined_samp, 'listing_id', do_scale=True)
yl = np.log(y)
df.head(40)
joined_testdf = joined_testdf.set_index("date")
df_test, _, nas, mapper = proc_df(joined_testdf, 'listing_id', do_scale=True,
                                  mapper=mapper, na_dict=nas)
Select the right time frame for validation data.

val_idx = np.flatnonzero(
    (df.index<=datetime.datetime(2018,8,31)) & (df.index>=datetime.datetime(2018,6,1)))
len(val_idx)
58817
m = RandomForestClassifier(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)
%time m.fit(df, y)
print_score(m)
DL
We can create a ModelData object directly from out data frame.

#md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y, cat_flds=cat_vars, bs=128, test_df=df_test)
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl, cat_flds=cat_vars, bs=128, test_df=None)
Some categorical variables have a lot more levels than others. listing_id, in particular, has over a thousand!

cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]
cat_sz
[('price', 558),
 ('bed_type', 6),
 ('host_is_superhost', 3),
 ('city', 60),
 ('state', 120),
 ('property_type', 45),
 ('room_type', 4),
 ('beds', 24),
 ('bedrooms', 14),
 ('bathrooms', 16),
 ('instant_bookable', 3),
 ('host_identity_verified', 3),
 ('host_total_listings_count', 119),
 ('calculated_host_listings_count', 82),
 ('is_business_travel_ready', 3),
 ('monthly_price', 524),
 ('minimum_nights', 68),
 ('maximum_nights', 186),
 ('review_scores_rating', 56),
 ('review_scores_accuracy', 10),
 ('review_scores_cleanliness', 10),
 ('review_scores_checkin', 10),
 ('review_scores_communication', 10),
 ('review_scores_location', 9),
 ('review_scores_value', 10),
 ('security_deposit', 282),
 ('cleaning_fee', 132),
 ('accommodates', 18),
 ('Year', 3),
 ('Month', 10),
 ('Week', 13),
 ('Day', 9),
 ('Dayofweek', 8)]
We use the cardinality of each variable (that is, its number of unique values) to decide how large to make its embeddings. Each level will be associated with a vector with length defined as below.

emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
emb_szs
[(558, 50),
 (6, 3),
 (3, 2),
 (60, 30),
 (120, 50),
 (45, 23),
 (4, 2),
 (24, 12),
 (14, 7),
 (16, 8),
 (3, 2),
 (3, 2),
 (119, 50),
 (82, 41),
 (3, 2),
 (524, 50),
 (68, 34),
 (186, 50),
 (56, 28),
 (10, 5),
 (10, 5),
 (10, 5),
 (10, 5),
 (9, 5),
 (10, 5),
 (282, 50),
 (132, 50),
 (18, 9),
 (3, 2),
 (10, 5),
 (13, 7),
 (9, 5),
 (8, 4)]
def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())
  
def logloss(y_pred, targ, eps=1e-15):
  p = np.clip(y_pred, eps, 1 - eps)
  if targ == 1:
    return -log(p)
  else:
    return -log(1 - p)

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
#m.summary()
lr = 1e-3
m.lr_find()
HBox(children=(IntProgress(value=0, description='Epoch', max=1, style=ProgressStyle(description_width='initial…
 59%|█████▉    | 721/1215 [00:18<00:11, 42.30it/s, loss=0.77] 
lr = 1e-3
m.fit(lr, 1, metrics=[exp_rmspe])
HBox(children=(IntProgress(value=0, description='Epoch', max=1, style=ProgressStyle(description_width='initial…
epoch      trn_loss   val_loss   exp_rmspe  
    0      0.029854   0.037996   0.211083  
[array([0.038]), 0.21108328053183206]
m.fit(lr, 3, metrics=[exp_rmspe])
HBox(children=(IntProgress(value=0, description='Epoch', max=3, style=ProgressStyle(description_width='initial…
epoch      trn_loss   val_loss   exp_rmspe  
    0      0.020831   0.030089   0.187496  
    1      0.014721   0.025881   0.161962  
    2      0.011954   0.024756   0.156278  
[array([0.02476]), 0.1562777694125062]
m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)
HBox(children=(IntProgress(value=0, description='Epoch', max=3, style=ProgressStyle(description_width='initial…
epoch      trn_loss   val_loss   exp_rmspe  
    0      0.006931   0.021426   0.143953  
    1      0.006298   0.021132   0.142215  
    2      0.006124   0.020785   0.140216  
[array([0.02078]), 0.14021607320871066]
Testing Time
??md.get_learner()
??
#md.get_learner(emb_szs, n_cont = len(df.columns)-len(cat_vars),emb_drop = 0.04, out_sz = 2, szs = [250,100], drops = [0.001,0.01], y_range=[0,1], use_bn = True)
Object `md.get_learner` not found.
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl, cat_flds=cat_vars, bs=128, test_df=df_test)
y_range = (0, max_log_y)
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   emb_drop = 0.04, out_sz = 1, szs = [1000,500], drops = [0.001,0.01], y_range=y_range)
lr = 1e-3
m.lr_find()
HBox(children=(IntProgress(value=0, description='Epoch', max=1, style=ProgressStyle(description_width='initial…
 65%|██████▍   | 789/1215 [00:17<00:08, 49.13it/s, loss=1.48]
lr = 1e-3
m.fit(lr, 1)
HBox(children=(IntProgress(value=0, description='Epoch', max=1, style=ProgressStyle(description_width='initial…
epoch      trn_loss   val_loss   
    0      0.285798   0.273232  

[array([0.27323])]
lr = 1e-3
m.fit(lr, 3)
HBox(children=(IntProgress(value=0, description='Epoch', max=3, style=ProgressStyle(description_width='initial…
epoch      trn_loss   val_loss   
    0      0.20793    0.244978  
    1      0.170314   0.190456  
    2      0.129314   0.194209  

[array([0.19421])]
m.fit(lr, 3)
HBox(children=(IntProgress(value=0, description='Epoch', max=3, style=ProgressStyle(description_width='initial…
epoch      trn_loss   val_loss   
    0      0.104495   0.172113  
    1      0.083297   0.147242  
    2      0.0718     0.131978  

[array([0.13198])]
m.save('val0')
m.load('val0')
joined_testdf.head()
x,y=m.predict_with_targs()
logloss(x,y)
exp_rmspe(x,y)
3.6758296367448238
pred_test = m.predict(True)
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-42-ee28ccb05714> in <module>()
----> 1 pred_test = m.predict(True)

/usr/local/lib/python3.6/dist-packages/fastai/learner.py in predict(self, is_test, use_swa)
    355         dl = self.data.test_dl if is_test else self.data.val_dl
    356         m = self.swa_model if use_swa else self.model
--> 357         return predict(m, dl)
    358 
    359     def predict_with_targs(self, is_test=False, use_swa=False):

/usr/local/lib/python3.6/dist-packages/fastai/model.py in predict(m, dl)
    220 def predict(m, dl):
    221     preda,_ = predict_with_targs_(m, dl)
--> 222     return to_np(torch.cat(preda))
    223 
    224 def predict_batch(m, x):

RuntimeError: cuda runtime error (59) : device-side assert triggered at /pytorch/torch/lib/THC/THCTensorCopy.cu:100
pred_test = np.exp(pred_test)
joined_testdf['listing_id']=pred_test
csv_fn=f'{PATH}tmp/sub.csv'
joined_testdf[['price','listing_id']].to_csv(csv_fn, index=False)
FileLink(csv_fn)
amsterdam/tmp/sub.csv
!cat ./amsterdam/tmp/sub.csv
price,listing_id
69.0,3626.0771
160.0,470576.66
80.0,16082.404
125.0,20006.154
150.0,24443.748
65.0,12389.733
75.0,18470.652
55.0,19432.041
130.0,33718.863
219.0,20727.303
115.0,40149.707
159.0,24677.201
210.0,38128.543
100.0,45301.484
250.0,66824.57
200.0,366349.53
150.0,95652.97
140.0,1884512.4
350.0,80976.19
420.0,50250.45
225.0,60224.1
120.0,86363.44
125.0,55076.75
110.0,87999.89
90.0,41690.562
78.0,21064.494
87.0,136733.69
75.0,34118.348
60.0,107561.625
86.0,41275.766
200.0,65483.15
250.0,201796.95
159.0,78832.88
60.0,42304.91
149.0,49172.375
112.0,2075376.9
225.0,178957.39
150.0,219923.06
109.0,81519.266
750.0,1139509.9
145.0,110023.37
450.0,68821.26
179.0,181047.19
109.0,61513.67
109.0,57269.5
70.0,132341.55
225.0,59521.22
75.0,70215.66
195.0,82450.7
260.0,34046.613
105.0,308178.78
70.0,217508.11
60.0,162601.05
135.0,148021.42
93.0,442849.44
95.0,35871.984
235.0,110318.39
45.0,110119.1
100.0,158253.8
189.0,168493.7
75.0,103894.8
65.0,89133.2
150.0,677359.7
100.0,128850.85
100.0,71922.805
165.0,88059.914
95.0,243240.02
87.0,48504.81
120.0,677724.1
99.0,263555.8
100.0,96708.44
80.0,110823.28
259.0,147393.3
98.0,99594.43
100.0,192367.5
80.0,151493.64
89.0,114994.586
695.0,315534.56
79.0,86578.01
145.0,289620.94
74.0,114362.67
120.0,2685865.2
220.0,187265.67
145.0,151582.67
120.0,149712.98

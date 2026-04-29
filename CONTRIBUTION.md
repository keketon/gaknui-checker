# CONTRIBUTION

## Requirements

* WSL2

## Assumption

* pyenv
* Python 3.11.9 in with venv

## Set up

### Install Python

Install tools that are required to build Python and related tools from source.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Install pyenv. DO FOLLOW steps prompted after the installation!!!

```bash
curl https://pyenv.run | bash
```

Install Python via pyenv and configure the project to use the fixed version

```bash
pyenv install 3.11.9

mkdir GaknuiCkr
cd GaknuiCkr
pyenv local 3.11.9

python --version
# 3.11.9 should be shown
```

Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure Playwright

```bash
playwright install chromium

# Install dependencies required to run on WSL
# `sudo playwright` doesn't work as it tries to use the globally-installed playwright, not the one in the venv.
sudo ./.venv/bin/python -m playwright install-deps
```

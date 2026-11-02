$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    py -3.12 -m venv (Join-Path $project '.venv')
    $venvPython = Join-Path $project '.venv\Scripts\python.exe'
}
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
& $venvPython -m pip install -r (Join-Path $project 'requirements.txt')
& $venvPython -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available())"

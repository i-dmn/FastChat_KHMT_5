param(
    [string]$ModelId = "meta-llama/llama-3.3-70b-instruct",
    [string]$ApiBaseUrl = "https://openrouter.ai/api/v1",
    [string]$CandidateName = "candidate",
    [string]$JudgeModel = "gpt-6-luna",
    [int]$QuestionCount = 4,
    [int]$Parallel = 2,
    [switch]$Full,
    [switch]$JudgeOnly,
    [string]$RunId,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$judgeDirectory = Join-Path $repoRoot "fastchat\llm_judge"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

function Invoke-Python {
    param([string[]]$Arguments)

    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Read-SecretAsPlainText {
    param([string]$Prompt)

    $secureValue = Read-Host -Prompt $Prompt -AsSecureString
    if ($secureValue.Length -eq 0) {
        throw "An API key is required."
    }

    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

if ($CandidateName -notmatch '^[A-Za-z0-9_-]+$') {
    throw "CandidateName must contain only letters, numbers, underscores, or hyphens."
}
if ($QuestionCount -lt 1) {
    throw "QuestionCount must be at least 1."
}
if ($Parallel -lt 1) {
    throw "Parallel must be at least 1."
}

if (-not (Test-Path $venvPython)) {
    if ($SkipInstall) {
        throw "The project virtual environment was not found. Run once without -SkipInstall."
    }

    Push-Location $repoRoot
    try {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.11 -m venv .venv
        }
        else {
            & python -m venv .venv
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the virtual environment. Install Python 3.11 and try again."
        }
    }
    finally {
        Pop-Location
    }
}

if (-not $SkipInstall) {
    Push-Location $repoRoot
    try {
        Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Python -Arguments @("-m", "pip", "install", "-e", ".[model_worker,llm_judge]")
        Invoke-Python -Arguments @("-m", "pip", "install", "pandas")
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path (Join-Path $judgeDirectory "data\mt_bench\question.jsonl"))) {
    throw "The bundled MT-Bench questions were not found under fastchat\llm_judge\data."
}

$sourceBenchmarkDirectory = Join-Path $judgeDirectory "data\mt_bench"
if ($JudgeOnly) {
    if ($RunId -notmatch '^run_[A-Za-z0-9_-]+$') {
        throw "For -JudgeOnly, pass the existing run folder name with -RunId (for example: run_20261109_160047)."
    }
    $runId = $RunId
}
else {
    $runId = "run_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

$benchName = "runs\$runId"
$runDataDirectory = Join-Path $judgeDirectory "data\$benchName"
$candidateAnswerDirectory = Join-Path $runDataDirectory "model_answer"
$referenceAnswerDirectory = Join-Path $runDataDirectory "reference_answer"
$questionFile = Join-Path $runDataDirectory "question.jsonl"
$answerFile = Join-Path $candidateAnswerDirectory "$CandidateName.jsonl"
$answerFileArgument = ".\data\$benchName\model_answer\$CandidateName.jsonl"

if ($JudgeOnly) {
    if (-not (Test-Path $questionFile) -or -not (Test-Path $answerFile) -or -not (Test-Path (Join-Path $referenceAnswerDirectory "gpt-4.jsonl"))) {
        throw "Run '$RunId' is missing its benchmark questions, candidate answers, or reference answers."
    }
}
else {
    New-Item -ItemType Directory -Force -Path $candidateAnswerDirectory, $referenceAnswerDirectory | Out-Null
    Copy-Item (Join-Path $sourceBenchmarkDirectory "question.jsonl") $questionFile
    Copy-Item (Join-Path $sourceBenchmarkDirectory "reference_answer\gpt-4.jsonl") (Join-Path $referenceAnswerDirectory "gpt-4.jsonl")

    $candidateApiKey = Read-SecretAsPlainText -Prompt "Candidate provider API key (hidden)"
    Push-Location $judgeDirectory
    try {
        $env:OPENAI_API_KEY = $candidateApiKey
        $answerArguments = @(
            ".\gen_api_answer.py",
            "--bench-name", $benchName,
            "--model", $ModelId,
            "--openai-api-base", $ApiBaseUrl,
            "--answer-file", $answerFileArgument,
            "--parallel", "$Parallel"
        )
        if (-not $Full) {
            $answerArguments += @("--question-end", "$QuestionCount")
        }
        Invoke-Python -Arguments $answerArguments
    }
    finally {
        Remove-Item Env:\OPENAI_API_KEY -ErrorAction SilentlyContinue
        $candidateApiKey = $null
        Pop-Location
    }
}

$judgeApiKey = Read-SecretAsPlainText -Prompt "OpenAI API key for the GPT-6 Luna judge (hidden)"
Push-Location $judgeDirectory
try {
    $env:OPENAI_API_KEY = $judgeApiKey
    $judgmentArguments = @(
        ".\gen_judgment.py",
        "--bench-name", $benchName,
        "--judge-file", ".\data\judge_prompts.jsonl",
        "--judge-model", $JudgeModel,
        "--model-list", $CandidateName,
        "--mode", "single",
        "--parallel", "$Parallel",
        "--yes"
    )
    if (-not $Full) {
        $judgmentArguments += @("--first-n", "$QuestionCount")
    }
    Invoke-Python -Arguments $judgmentArguments
    Invoke-Python -Arguments @(
        ".\show_result.py",
        "--bench-name", $benchName,
        "--judge-model", $JudgeModel,
        "--model-list", $CandidateName
    )
}
finally {
    Remove-Item Env:\OPENAI_API_KEY -ErrorAction SilentlyContinue
    $judgeApiKey = $null
    Pop-Location
}

Write-Host "Run complete. Answers, judgments, and scores are in $runDataDirectory"

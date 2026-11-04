$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try { & '.\.venv\Scripts\python.exe' -X utf8 -m voicebridge.gui } finally { Pop-Location }

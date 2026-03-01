# Align backend – build and run (PowerShell)
# Usage: .\build.ps1 [install | db-setup | run | mysql]

param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "db-setup", "run", "mysql", "help", "")]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
$HostPort = "127.0.0.1:8000"

# Load .env if present (simple key=value, no quotes)
function Load-Env {
    $envPath = Join-Path $PSScriptRoot ".env"
    if (Test-Path $envPath) {
        Get-Content $envPath | ForEach-Object {
            if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
                [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
            }
        }
    }
}

# Resolve mysql.exe: use MYSQL_CMD, else common install paths, else "mysql" (PATH)
function Get-MysqlExe {
    if ($env:MYSQL_CMD -and (Test-Path $env:MYSQL_CMD)) { return $env:MYSQL_CMD }
    $paths = @(
        "C:\Program Files\MySQL\MySQL Server 8.4\bin\mysql.exe",
        "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe",
        "C:\Program Files\MySQL\MySQL Server 9.0\bin\mysql.exe"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) { return $p }
    }
    return "mysql"
}

function Show-Help {
    Write-Host "Targets:"
    Write-Host "  .\build.ps1 install   - Install Python dependencies"
    Write-Host "  .\build.ps1 db-setup  - Create DB and tables (uses MYSQL_* env or .env)"
    Write-Host "  .\build.ps1 run       - Start FastAPI server (uvicorn)"
    Write-Host "  .\build.ps1 mysql     - Open MySQL CLI for MYSQL_DATABASE"
    Write-Host ""
    Write-Host "If 'mysql' is not on PATH, set MYSQL_CMD in .env to the full path to mysql.exe,"
    Write-Host "  e.g. MYSQL_CMD=C:\Program Files\MySQL\MySQL Server 8.4\bin\mysql.exe"
    Write-Host "  Or the script will try common MySQL Server install paths."
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\build.ps1 install"
    Write-Host "  .\build.ps1 db-setup"
    Write-Host "  .\build.ps1 run"
}

function Invoke-Install {
    pip install -r requirements.txt
}

function Invoke-DbSetup {
    Load-Env
    $mysqlExe = Get-MysqlExe
    $host_ = if ($env:MYSQL_HOST) { $env:MYSQL_HOST } else { "localhost" }
    $user = if ($env:MYSQL_USER) { $env:MYSQL_USER } else { "root" }
    $pass = if ($env:MYSQL_PASSWORD) { $env:MYSQL_PASSWORD } else { "" }
    $schema = (Resolve-Path (Join-Path $PSScriptRoot "schema.sql")).Path -replace "\\", "/"
    if (-not (Test-Path $schema)) { throw "schema.sql not found" }
    Write-Host "Applying schema.sql..."
    $args = @("-h", $host_, "-u", $user, "-e", "source $schema")
    if ($pass) { $args = @("-h", $host_, "-u", $user, "-p$pass", "-e", "source $schema") }
    & $mysqlExe @args
    Write-Host "Database ready."
}

function Invoke-Run {
    Push-Location $PSScriptRoot
    try {
        python -m uvicorn routes:app --host 127.0.0.1 --port 8000 --reload
    } finally {
        Pop-Location
    }
}

function Invoke-Mysql {
    Load-Env
    $mysqlExe = Get-MysqlExe
    $host_ = if ($env:MYSQL_HOST) { $env:MYSQL_HOST } else { "localhost" }
    $user = if ($env:MYSQL_USER) { $env:MYSQL_USER } else { "root" }
    $pass = if ($env:MYSQL_PASSWORD) { $env:MYSQL_PASSWORD } else { "" }
    $db   = if ($env:MYSQL_DATABASE) { $env:MYSQL_DATABASE } else { "align" }
    $args = @("-h", $host_, "-u", $user, $db)
    if ($pass) { $args = @("-h", $host_, "-u", $user, "-p$pass", $db) }
    & $mysqlExe @args
}

Load-Env

switch ($Target) {
    "install"  { Invoke-Install }
    "db-setup" { Invoke-DbSetup }
    "run"      { Invoke-Run }
    "mysql"    { Invoke-Mysql }
    default    { Show-Help }
}

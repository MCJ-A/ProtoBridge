[CmdletBinding()]
param(
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

$ScriptParent = Split-Path -Parent $PSScriptRoot
$RepoRoot = $null
if (Test-Path (Join-Path $ScriptParent "Ignait.exe")) {
    $SourceDir = $ScriptParent
    $PossibleRepo = Split-Path -Parent (Split-Path -Parent $ScriptParent)
    if (Test-Path (Join-Path $PossibleRepo "ignait\__init__.py")) {
        $RepoRoot = $PossibleRepo
    }
} elseif (Test-Path (Join-Path $ScriptParent "dist\Ignait\Ignait.exe")) {
    $SourceDir = Join-Path $ScriptParent "dist\Ignait"
    $RepoRoot = $ScriptParent
} else {
    throw "No se encontro Ignait.exe en '$ScriptParent' ni en '$ScriptParent\dist\Ignait'. Compila primero con PyInstaller (Ignait.spec)."
}

$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\Ignait"
$IconSrc = @(
    (Join-Path $SourceDir "_internal\ignait\assets\ignait.ico"),
    (Join-Path $SourceDir "ignait.ico"),
    $(if ($RepoRoot) { Join-Path $RepoRoot "ignait\assets\ignait.ico" } else { "" })
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $IconSrc) {
    throw "No se encontro ignait.ico para completar la instalacion."
}

Write-Host "[1/5] Deteniendo instancias previas de Ignait Studio..." -ForegroundColor Cyan
Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and (
        $_.Name -eq "Ignait.exe" -or
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase)) -or
        ($_.CommandLine -like "*xtts_worker.py*" -and $_.CommandLine -like "*Programs\Ignait*")
    )
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 400

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

# Desmontar junctions previos de forma segura SIN borrar el contenido real
foreach ($jName in @("models", ".venv-media-tts")) {
    $jPath = Join-Path $InstallDir $jName
    $item = Get-Item $jPath -Force -ErrorAction SilentlyContinue
    if ($item) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            cmd /c rmdir "`"$jPath`""
            if ($LASTEXITCODE -ne 0) { throw "No se pudo desmontar el enlace previo '$jPath'." }
        }
    }
}

Write-Host "[2/5] Copiando binarios de Ignait Studio a '$InstallDir'..." -ForegroundColor Cyan
Copy-Item (Join-Path $SourceDir "Ignait.exe") (Join-Path $InstallDir "Ignait.exe") -Force
Copy-Item $IconSrc (Join-Path $InstallDir "ignait.ico") -Force

$DstInternal = Join-Path $InstallDir "_internal"
if (Test-Path $DstInternal) { Remove-Item $DstInternal -Recurse -Force }
Copy-Item (Join-Path $SourceDir "_internal") $DstInternal -Recurse -Force

$SrcExt = Join-Path $SourceDir "browser_extension"
$DstExt = Join-Path $InstallDir "browser_extension"
if (Test-Path $SrcExt) {
    if (Test-Path $DstExt) { Remove-Item $DstExt -Recurse -Force }
    Copy-Item $SrcExt $DstExt -Recurse -Force
}

$DstConfig = Join-Path $InstallDir "config.json"
$SrcConfig = Join-Path $SourceDir "config.json"
if (-not (Test-Path $DstConfig) -and (Test-Path $SrcConfig)) {
    Copy-Item $SrcConfig $DstConfig -Force
}

Write-Host "[3/5] Configurando modelos IA y sidecar XTTS-v2..." -ForegroundColor Cyan
$ModelsDst = Join-Path $InstallDir "models"
$ModelsCandidates = @()
if ($RepoRoot) { $ModelsCandidates += (Join-Path $RepoRoot "models") }
$ModelsCandidates += (Join-Path $SourceDir "models")

$ModelsSrc = $null
foreach ($cand in $ModelsCandidates) {
    if (Test-Path $cand) {
        $itemCand = Get-Item $cand -Force
        $resolvedCand = $itemCand.Target | Select-Object -First 1
        if (-not $resolvedCand) { $resolvedCand = (Resolve-Path $cand).Path }
        $hasFiles = Get-ChildItem $resolvedCand -Force -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hasFiles) {
            $ModelsSrc = $resolvedCand
            break
        }
    }
}

if ($ModelsSrc) {
    if (-not (Test-Path $ModelsDst)) {
        if ([IO.Path]::GetPathRoot($ModelsSrc) -eq [IO.Path]::GetPathRoot($InstallDir)) {
            cmd /c mklink /J "`"$ModelsDst`"" "`"$ModelsSrc`"" | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "No se pudo crear junction a models en '$ModelsDst'." }
        } else {
            Copy-Item $ModelsSrc $ModelsDst -Recurse -Force
        }
    }
} elseif (-not (Test-Path $ModelsDst)) {
    New-Item -ItemType Directory -Path $ModelsDst -Force | Out-Null
}

$VenvTtsDst = Join-Path $InstallDir ".venv-media-tts"
$VenvTtsCandidates = @()
if ($RepoRoot) { $VenvTtsCandidates += (Join-Path $RepoRoot ".venv-media-tts") }
$VenvTtsCandidates += (Join-Path $SourceDir ".venv-media-tts")
$VenvTtsSrc = $null
foreach ($cand in $VenvTtsCandidates) {
    if (Test-Path (Join-Path $cand "Scripts\python.exe")) {
        $itemV = Get-Item $cand -Force
        $resolvedV = $itemV.Target | Select-Object -First 1
        if (-not $resolvedV) { $resolvedV = (Resolve-Path $cand).Path }
        $VenvTtsSrc = $resolvedV
        break
    }
}
if ($VenvTtsSrc -and -not (Test-Path $VenvTtsDst)) {
    if ([IO.Path]::GetPathRoot($VenvTtsSrc) -eq [IO.Path]::GetPathRoot($InstallDir)) {
        cmd /c mklink /J "`"$VenvTtsDst`"" "`"$VenvTtsSrc`"" | Out-Null
    }
}

Write-Host "[4/5] Creando accesos directos y desinstalador..." -ForegroundColor Cyan
$UninstallPs1 = Join-Path $InstallDir "Uninstall_Ignait.ps1"
$UninstallContent = @'
$ErrorActionPreference = "SilentlyContinue"
$InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -and $_.ExecutablePath.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase)
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Milliseconds 400

foreach ($jName in @("models", ".venv-media-tts")) {
    $jPath = Join-Path $InstallDir $jName
    $item = Get-Item $jPath -Force
    if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        cmd /c rmdir "`"$jPath`"" | Out-Null
    }
}

$DesktopLnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Ignait Studio.lnk"
$StartMenuLnk = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Ignait Studio.lnk"
if (Test-Path $DesktopLnk) { Remove-Item $DesktopLnk -Force }
if (Test-Path $StartMenuLnk) { Remove-Item $StartMenuLnk -Force }

$RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\IgnaitStudio"
if (Test-Path $RegPath) { Remove-Item $RegPath -Recurse -Force }

Start-Process cmd.exe -ArgumentList "/c ping 127.0.0.1 -n 2 >nul & rmdir /s /q `"$InstallDir`"" -WindowStyle Hidden
'@
[IO.File]::WriteAllText($UninstallPs1, $UninstallContent, [System.Text.UTF8Encoding]::new($false))

$UninstallBat = Join-Path $InstallDir "Uninstall_Ignait.bat"
$UninstallBatContent = "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0Uninstall_Ignait.ps1`"`r`n"
[IO.File]::WriteAllText($UninstallBat, $UninstallBatContent, [System.Text.Encoding]::ASCII)

$ExeInstalled = Join-Path $InstallDir "Ignait.exe"
$IcoInstalled = Join-Path $InstallDir "ignait.ico"
$DesktopDir = [Environment]::GetFolderPath("Desktop")
$StartMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
New-Item -ItemType Directory -Path $StartMenuDir -Force | Out-Null

$LegacyDesktopLnk = Join-Path $DesktopDir "ProtoBridge Studio (by Cresko).lnk"
if (Test-Path $LegacyDesktopLnk) { Remove-Item $LegacyDesktopLnk -Force -ErrorAction SilentlyContinue }

$Shell = New-Object -ComObject WScript.Shell
foreach ($lnkPath in @(
    (Join-Path $DesktopDir "Ignait Studio.lnk"),
    (Join-Path $StartMenuDir "Ignait Studio.lnk")
)) {
    $sc = $Shell.CreateShortcut($lnkPath)
    $sc.TargetPath = $ExeInstalled
    $sc.WorkingDirectory = $InstallDir
    $sc.IconLocation = "$IcoInstalled,0"
    $sc.Description = "Ignait Studio - Real-Time Voice & Media Translator (Cresko · VE)"
    $sc.Save()
}

Write-Host "[5/5] Registrando en Aplicaciones instaladas de Windows..." -ForegroundColor Cyan
$RegKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\IgnaitStudio"
New-Item -Path $RegKey -Force | Out-Null
Set-ItemProperty -Path $RegKey -Name "DisplayName" -Value "Ignait Studio"
Set-ItemProperty -Path $RegKey -Name "DisplayVersion" -Value "1.0.0"
Set-ItemProperty -Path $RegKey -Name "Publisher" -Value "Jhon Córdova Martínez (Cresko · VE)"
Set-ItemProperty -Path $RegKey -Name "DisplayIcon" -Value $IcoInstalled
Set-ItemProperty -Path $RegKey -Name "InstallLocation" -Value $InstallDir
Set-ItemProperty -Path $RegKey -Name "UninstallString" -Value "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$UninstallPs1`""
Set-ItemProperty -Path $RegKey -Name "QuietUninstallString" -Value "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$UninstallPs1`""
Set-ItemProperty -Path $RegKey -Name "NoModify" -Value 1 -Type DWord
Set-ItemProperty -Path $RegKey -Name "NoRepair" -Value 1 -Type DWord

Write-Host "INSTALL_OK dir='$InstallDir' models_exists=$(Test-Path $ModelsDst)" -ForegroundColor Green

if (-not $NoLaunch) {
    Start-Process -FilePath $ExeInstalled -WorkingDirectory $InstallDir
}

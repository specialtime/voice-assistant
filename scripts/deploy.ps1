# deploy.ps1 — Deploya dev → prod (copia solo código + scripts, sin configs/logs/docs).
# Uso: powershell -ExecutionPolicy Bypass -File deploy.ps1
# Requiere: $env:CORTEX_PROD_DIR apuntando al dir de prod (default: C:\Users\crist\voice-assistant)

$ErrorActionPreference = "Stop"

$DEV_DIR = $PSScriptRoot
$PROD_DIR = if ($env:CORTEX_PROD_DIR) { $env:CORTEX_PROD_DIR } else { "C:\Users\crist\voice-assistant" }

Write-Host "[INFO] Deploy dev → prod"
Write-Host "[INFO] Dev:  $DEV_DIR"
Write-Host "[INFO] Prod: $PROD_DIR"

# Verificar que prod existe
if (-not (Test-Path -LiteralPath $PROD_DIR)) {
    Write-Host "[ERROR] Directorio de prod no existe: $PROD_DIR"
    exit 1
}

# Verificar que dev tiene src\
if (-not (Test-Path -LiteralPath "$DEV_DIR\src")) {
    Write-Host "[ERROR] Directorio de dev no tiene src\: $DEV_DIR"
    exit 1
}

# Archivos/dirs a copiar (relativos a DEV_DIR)
# NOTA: NO se copia config\settings.json porque dev y prod tienen configs distintas
#       (dev: logging.level=DEBUG, prod: logging.level=INFO). Si el código necesita
#       un nuevo campo de config, agregarlo manualmente a prod.
# NOTA: NO se copia .env.example ni .gitignore (prod no los necesita).
$toCopy = @(
    "src",
    "requirements.txt",
    "start_cortex.bat",
    "start_opencode_hidden.py"
)

foreach ($item in $toCopy) {
    $src = "$DEV_DIR\$item"
    $dst = "$PROD_DIR\$item"
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Host "[WARN] No existe en dev, se salta: $item"
        continue
    }
    if (Test-Path -LiteralPath $src -PathType Container) {
        # Es un directorio (src\) — eliminar destino si existe, luego copiar
        # FIX (2026-06-22): Copy-Item anida el dir dentro del destino si este ya existe
        # (crea src\src\ en vez de sobreescribir src\). Solución: eliminar destino primero.
        if (Test-Path -LiteralPath $dst) {
            Remove-Item -LiteralPath $dst -Recurse -Force
        }
        Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
    } else {
        # Es un archivo
        $dstDir = Split-Path -Parent $dst
        if (-not (Test-Path -LiteralPath $dstDir)) {
            New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        }
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
    Write-Host "[OK] Copiado: $item"
}

# Limpiar __pycache__ en prod (post-copia)
Get-ChildItem -Path $PROD_DIR -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "[OK] Deploy completado."
Write-Host "[INFO] Si requirements.txt cambió, ejecutá: pip install -r requirements.txt"
Write-Host "[INFO] NO se copió config\settings.json — si el código necesita un nuevo campo,"
Write-Host "       agregalo manualmente a $PROD_DIR\config\settings.json"

# stop-cortex.ps1 — Detiene el asistente de voz (dev o prod).
# Mata opencode.exe (por puerto 57214/57215) + pythonw.exe del asistente (por línea de comandos).
# Uso: powershell -ExecutionPolicy Bypass -File stop-cortex.ps1

$stopped = 0

# 1. Matar opencode.exe que escucha en los puertos del asistente (57214 prod, 57215 dev)
foreach ($port in @(57214, 57215)) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($conn in $conns) {
            $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($proc) {
                Stop-Process -Id $proc.Id -Force
                Write-Host "[OK] opencode.exe detenido (pid=$($proc.Id), puerto=$port)"
                $stopped++
            }
        }
    }
}

# 2. Matar pythonw.exe del asistente (wrapper start_opencode_hidden.py + orquestador src\main.py)
$pythonwProcs = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue
foreach ($proc in $pythonwProcs) {
    if ($proc.CommandLine -match 'start_opencode_hidden\.py' -or $proc.CommandLine -match 'src[\\/]main\.py') {
        Stop-Process -Id $proc.ProcessId -Force
        Write-Host "[OK] pythonw.exe detenido (pid=$($proc.ProcessId), cmd=$($proc.CommandLine.Substring(0, [Math]::Min(80, $proc.CommandLine.Length))))"
        $stopped++
    }
}

if ($stopped -eq 0) {
    Write-Host "[INFO] No hay procesos del asistente corriendo."
} else {
    Write-Host "[OK] Total procesos detenidos: $stopped"
}

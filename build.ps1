$vsPath = "C:\Program Files\Microsoft Visual Studio\18\Community"
$vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"

if (-not (Test-Path $vcvarsPath)) {
    Write-Error "Could not find vcvarsall.bat at $vcvarsPath"
    exit 1
}

Write-Host "Setting up VS environment..."
& "C:\Windows\System32\cmd.exe" /c "call `"$vcvarsPath`" x64 && set > env.txt"

# Load environment variables from the temp file
Get-Content env.txt | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Path "env:\$($matches[1])" -Value $matches[2]
    }
}
Remove-Item env.txt

Write-Host "Running CMake configure..."
cmake --preset x64-relwithdebinfo

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configure failed"
    exit $LASTEXITCODE
}

Write-Host "Building project..."
cmake --build out/build/x64-relwithdebinfo

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    exit $LASTEXITCODE
}

Write-Host "Build successful!"

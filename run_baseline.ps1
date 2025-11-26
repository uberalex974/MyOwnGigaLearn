$process = Start-Process -FilePath "out\build\x64-relwithdebinfo\GigaLearnBot.exe" -RedirectStandardOutput "baseline_log.txt" -PassThru
Start-Sleep -Seconds 70
Stop-Process -Id $process.Id
Get-Content "baseline_log.txt" -Tail 50

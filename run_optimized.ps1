$process = Start-Process -FilePath "out\build\x64-relwithdebinfo\GigaLearnBot.exe" -RedirectStandardOutput "optimized_log.txt" -PassThru
Start-Sleep -Seconds 70
Stop-Process -Id $process.Id
Get-Content "optimized_log.txt" -Tail 50

' run_whisper_stt.vbs - launches whisper_voice_type.py silently (no console window).
' Used both for manual run (double-click) and from the Startup folder shortcut.
'
' Resolves the repo dir as the folder this .vbs lives in, so it works no matter
' where you cloned bidet-quick. Uses the .venv\Scripts\pythonw.exe if present,
' otherwise falls back to the system pythonw.exe on PATH.

Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoDir = fso.GetParentFolderName(WScript.ScriptFullName)
script = repoDir & "\whisper_voice_type.py"
venvPy = repoDir & "\.venv\Scripts\pythonw.exe"

If fso.FileExists(venvPy) Then
    py = venvPy
Else
    py = "pythonw.exe"   ' fall back to whatever pythonw is on PATH
End If

sh.CurrentDirectory = repoDir
sh.Run """" & py & """ """ & script & """", 0, False

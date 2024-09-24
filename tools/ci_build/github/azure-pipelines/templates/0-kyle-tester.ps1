
$working_directory = $Args[0]
$jar_file_directory = $working_directory + "\debugging_target_folder"
New-Item -ItemType "directory" -Path $jar_file_directory

Write-Host "Generating fake jar files."
Out-File -FilePath $jar_file_directory"\test1.jar" -InputObject "this is a test1 jar file."
Write-Host "Generated fake jar files."

$gpg_path = "C:\Program Files (x86)\gnupg\bin\gpg.exe"
$passphrase_file = Join-Path -Path $working_directory -ChildPath "passphrase.txt"
$private_key_file = Join-Path -Path $working_directory -ChildPath "private_key.txt"


#format 1
try {
    Write-Host "Running Format 1......."
    Write-Host "Generating passphrase and private key files."h
    Out-File -FilePath $passphrase_file -InputObject $(java-pgp-pwd) -NoNewline -Encoding ascii
    Out-File -FilePath $private_key_file -InputObject $(java-pgp-key) -NoNewline -Encoding ascii

    Write-Host "==========pwd.length: " + $(java-pgp-pwd).Length
    Write-Host "==========key.length: " + $(java-pgp-key).Length
    [string[]]$key_lines = Get-Content -Path $private_key_file
    Write-Host "==========Key file, lines count:"$key_lines.Count

    Write-Host "Generated passphrase and private key files."

    Write-Host "Importing private key file."
    $import_key_args_list = "--batch --import `"$private_key_file`""
    Start-Process -FilePath $GPG_PATH -ArgumentList $import_key_args_list -NoNewWindow -PassThru -Wait
    Write-Host "Imported private key file."
    Write-Host "Format 1 completed."
}
catch {
    Write-Host "FAILED: format 1"
}


#format 3
try {
    Write-Host "Running Format 3......."
    Write-Host "Generating passphrase and private key files."
    $pwd_value = '$(java-pgp-pwd)'
    $key_value = '$(java-pgp-key)'
    Out-File -FilePath $passphrase_file -InputObject $pwd_value -NoNewline -Encoding ascii
    Out-File -FilePath $private_key_file -InputObject $key_value -NoNewline -Encoding ascii

    Write-Host "==========pwd.length: " + $pwd.Length
    Write-Host "==========key.length: " + $key.Length
    [string[]]$key_lines = Get-Content -Path $private_key_file
    Write-Host "==========Key file, lines count:"$key_lines.Count

    Write-Host "Generated passphrase and private key files."


    Write-Host "Importing private key file."
    $import_key_args_list = "--batch --import `"$private_key_file`""
    Start-Process -FilePath $GPG_PATH -ArgumentList $import_key_args_list -NoNewWindow -PassThru -Wait
    Write-Host "Imported private key file."
    Write-Host "Format 3 completed."
}
catch {
    Write-Host "FAILED: format 3"
}

Write-Host "GPG and sha256 signing to files completed."
Write-Host "Deleting passphrase and private key files."
Remove-Item -Path $passphrase_file
Remove-Item -Path $private_key_file
Write-Host "Deleted passphrase and private key files."


$working_directory = $Args[0]
$original_passphrase = '$(java-pgp-pwd)'
$original_private_key = '$(java-pgp-key)'

$jar_file_directory = $working_directory + "\debugging_target_folder"
New-Item -ItemType "directory" -Path $jar_file_directory

Write-Host "Generating fake jar files."
Out-File -FilePath $jar_file_directory"\test1.jar" -InputObject "this is a test1 jar file."
Write-Host "Generated fake jar files."

$gpg_path = "C:\Program Files (x86)\gnupg\bin\gpg.exe"
$passphrase_file = Join-Path -Path $working_directory -ChildPath "passphrase.txt"
$private_key_file = Join-Path -Path $working_directory -ChildPath "private_key.txt"


Write-Host "Generating passphrase and private key files."
Out-File -FilePath $passphrase_file -InputObject $original_passphrase -NoNewline -Encoding ascii
Out-File -FilePath $private_key_file -InputObject $original_private_key -NoNewline -Encoding ascii
Write-Host "Generated passphrase and private key files."

[string[]]$key_lines = Get-Content -Path $private_key_file
Write-Host "===========Key lines count:"$key_lines.Count

Write-Host "Importing private key file."
$import_key_args_list = "--batch --import `"$private_key_file`""
Start-Process -FilePath $GPG_PATH -ArgumentList $import_key_args_list -NoNewWindow -PassThru -Wait
Write-Host "Imported private key file."

$targeting_original_files = Get-ChildItem $jar_file_directory -Recurse -Force -File -Name
foreach ($file in $targeting_original_files) {
    $file_path = Join-Path $jar_file_directory -ChildPath $file
    Write-Host "GPG signing to file: "$file_path
    $args_list = "--pinentry-mode loopback --passphrase-file `"$passphrase_file`" -ab `"$file_path`""
    Start-Process -FilePath $GPG_PATH -ArgumentList $args_list -NoNewWindow -PassThru -Wait
}

$targeting_asc_files = Get-ChildItem $jar_file_directory -Recurse -Force -File -Name
foreach ($file in $targeting_asc_files) {
    $file_path = Join-Path $jar_file_directory -ChildPath $file
    Write-Host "Adding checksum of sha256 to file: "$file_path
    $file_path_sha256 = $file_path + ".sha256"
    CertUtil -hashfile $file_path SHA256
    CertUtil -hashfile $file_path SHA256 | find /v `"hash`" | Out-File -FilePath $file_path_sha256
}

Write-Host "GPG and sha256 signing to files completed."
Write-Host "Deleting passphrase and private key files."
Remove-Item -Path $passphrase_file
Remove-Item -Path $private_key_file
Write-Host "Deleted passphrase and private key files."

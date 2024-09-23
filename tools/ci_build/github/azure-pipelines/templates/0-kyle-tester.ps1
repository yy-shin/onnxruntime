
Write-Host "================GPG VERSION START================="
$GPG_PATH = "C:\Program Files (x86)\gnupg\bin\gpg.exe"
Start-Process -FilePath $GPG_PATH -ArgumentList "--version" -NoNewWindow -PassThru -Wait
Write-Host "================GPG VERSION END================="

Write-Host "================PARAM EXISTED START================="
if (${env:java-pgp-pwd}) {
    Write-Host "YES! java-pgp-pwd exists"
}
else {
    Write-Host "NO! java-pgp-pwd NOT exists"
}

if (${env:java-pgp-key}) {
    Write-Host "YES! java-pgp-key exists"
}
else {
    Write-Host "NO! java-pgp-key NOT exists"
}

Write-Host "================PARAM EXISTED START================="
exit 1


Write-Host "================GPG VERSION START================="
$GPG_PATH = "C:\Program Files (x86)\gnupg\bin\gpg.exe"
Start-Process -FilePath $GPG_PATH -ArgumentList "--version" -NoNewWindow -PassThru -Wait
Write-Host "================GPG VERSION END================="

Write-Host "================PARAM EXISTED START - env:XXX ================="
if (${env:java-pgp-pwd}) {
    Write-Host "YES! java-pgp-pwd exists - env:XXX "
}
else {
    Write-Host "NO! java-pgp-pwd NOT exists - env:XXX "
}

if (${env:java-pgp-key}) {
    Write-Host "YES! java-pgp-key exists - env:XXX "
}
else {
    Write-Host "NO! java-pgp-key NOT exists - env:XXX "
}

Write-Host "================PARAM EXISTED START - env:XXX ================="

Write-Host "================PARAM EXISTED START - parameters.XXX ================="
if (${{ parameters.java-pgp-pwd }}) {
    Write-Host "YES! java-pgp-pwd exists - parameters.XXX "
}
else {
    Write-Host "NO! java-pgp-pwd NOT exists - parameters.XXX "
}

if (${{ parameters.java-pgp-key }}) {
    Write-Host "YES! java-pgp-key exists - parameters.XXX "
}
else {
    Write-Host "NO! java-pgp-key NOT exists - parameters.XXX "
}

Write-Host "================PARAM EXISTED START - parameters.XXX ================="

exit 1

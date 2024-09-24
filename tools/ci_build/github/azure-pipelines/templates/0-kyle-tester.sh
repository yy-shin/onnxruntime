#!/bin/bash
working_directory=$(Build.BinariesDirectory)
original_private_key=$(java-pgp-key)
original_passphrase=$(java-pgp-pwd)

#working_directory=$(pwd)
#original_private_key=$GPG_KEY
#original_passphrase=$GPG_PWD

size_of_private_key=${#original_private_key}
size_of_passphrase=${#original_passphrase}
echo "Size of private key: $size_of_private_key"
echo "Size of passphrase: $size_of_passphrase"

echo "Verifying GPG installation"
gpg --version

jar_files_directory=$working_directory/debugging_target_folder
mkdir -p $jar_files_directory
echo "this is a testing jar file." >$jar_files_directory/testing.jar

printf "%s" "$original_private_key" >$working_directory/private_key.txt
printf "%s" "$original_passphrase" >$working_directory/passphrase.txt

readarray -t private_key_array < $working_directory/private_key.txt
readarray -t passphrase_array < $working_directory/passphrase.txt
echo "private key length: ${#private_key_array[@]}"
echo "passphrase length: ${#passphrase_array[@]}"

echo "Importing GPG key"
gpg --batch --import $working_directory/private_key.txt
echo "Importing GPG key done"

echo "Signing jar file"
gpg --pinentry-mode loopback --passphrase-file $working_directory/passphrase.txt -ab $jar_files_directory/testing.jar
echo "Signing jar file done"

echo "generating checksum sha256"
sha256sum $jar_files_directory/testing.jar | awk '{print $1}' >$jar_files_directory/testing.jar.sha256

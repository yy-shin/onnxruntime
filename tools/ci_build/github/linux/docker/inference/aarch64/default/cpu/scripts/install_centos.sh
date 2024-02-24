#!/bin/bash
set -e -x

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)
echo "installing for CentOS version : $os_major_version"
dnf config-manager --set-disabled "ubi-$os_major_version-codeready-builder-rpms"
dnf config-manager  --save --setopt ubi-8-appstream-rpms.exclude=dotnet*,aspnetcore*,netstandard*
rpm -Uvh https://packages.microsoft.com/config/centos/$os_major_version/packages-microsoft-prod.rpm
dnf install -y python3.11-devel glibc-langpack-\* glibc-locale-source which redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel graphviz gcc-toolset-12-binutils gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ gcc-toolset-12-libasan-devel git
locale
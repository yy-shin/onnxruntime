#!/bin/bash
set -e -x

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)
echo "installing for CentOS version : $os_major_version"
dnf config-manager --set-disabled "ubi-$os_major_version-codeready-builder-rpms"
dnf config-manager  --save --setopt ubi-8-appstream-rpms.exclude=dotnet*,aspnetcore*,netstandard*
rpm -Uvh https://packages.microsoft.com/config/centos/$os_major_version/packages-microsoft-prod.rpm
dnf install -y python3.11-devel glibc-langpack-\* glibc-locale-source which redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel msopenjdk-11 graphviz gcc-toolset-12-binutils gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ gcc-toolset-12-libasan-devel dotnet-sdk-8.0 git
locale
dotnet --list-sdks
#For running C# tests
#TODO: "dotnet test /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp.csproj" command reports "The target platform identifier ios was not recognized". But ios workload is not available on Linux
#TODO: The workload 'android' is out of support and will not receive security updates in the future
dotnet workload install android wasm-tools-net6
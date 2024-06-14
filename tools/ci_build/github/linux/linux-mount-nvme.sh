#!/bin/bash 
# reference: https://techcommunity.microsoft.com/t5/azure-high-performance-computing/getting-started-with-the-nc-a100-v4-series/ba-p/3568843

set -ex

NVME_DISKS_NAME=$( ls /dev/nvme*n1 )
NVME_DISKS=$( ls -latr /dev/nvme*n1 | wc -l )
MOUNT_POINT="/nvme_disk"

echo "Number of NVMe Disks: $NVME_DISKS"

if [ "$NVME_DISKS" == "0" ]
then
    exit 0
else
    mkdir -p $MOUNT_POINT
    # Needed incase something did not unmount as expected. This will delete any data that may be left behind
    # mdadm  --stop /dev/md*
    mdadm --create /dev/md128 -f --run --level 0 --raid-devices $NVME_DISKS $NVME_DISKS_NAME
    mkfs.xfs -f /dev/md128
    mount /dev/md128 $MOUNT_POINT
fi

chmod 1777 $MOUNT_POINT

df -h
cat /etc/fstab
blkid

NVME_UUID=$(blkid | grep "/dev/md128" | grep -o 'UUID="[a-f0-9-]\+"' | head -n 1 | awk -F'=' '{print $2}' | tr -d '"')
echo "UUID=$NVME_UUID  $MOUNT_POINT   xfs   defaults,nofail   0   2" >> /etc/fstab

cat /etc/fstab

exit 0
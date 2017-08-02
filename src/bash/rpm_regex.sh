#!/bin/bash
function error
{
    echo "Error: $1"
    exit 1
}
function dumpToProducts
{
cat <<EOF > products
%asset $ASSET
%version $VERSION
%packaging $PACKAGE
%bundle $RPM_FILE $BUNDLE
%bundle $RPM_DEVEL_FILE ${BUNDLE}-devel
EOF
}
which UploadAsset >/dev/null 2>&1
[ $? -ne 0 ] && error "Cannot find UploadAsset"

ASSET="GEMF"
BUNDLE="libgemdecoder"
VERSION=""
RPM_FILE=""
RPM_DEVEL_FILE=""
PACKAGE="x86_64.rpm"

devel_num=0
rpm_num=0
devel_release=""
rmp_release=""
for f in $(find . -name "*.rpm");do
    if [[ $f == *"-devel-"* ]]; then
        devel_num=$(($devel_num + 1))
        RPM_DEVEL_FILE=${f#./}
        VERSION=$(echo $RPM_DEVEL_FILE | grep -oE "([0-9]+\.){2}[0-9]+(-[0-9]+)?")
    else
        rpm_num=$(($rpm_num + 1))
        RPM_FILE=${f#./}
        VERSION1=$(echo $RPM_FILE | grep -oE "([0-9]+\.){2}[0-9]+(-[0-9]+)?")
    fi
done
[ $devel_num -ne 1 ] && error "Devel RPM files not generated as expected"
[ $rpm_num -ne 1 ] && error "RPM files not generated as expected"
[ -z "$VERSION" ] && error "VERSION not as expected"
[ "$VERSION" != "$VERSION1" ] && error "VERSION damaged"

dumpToProducts
[ $? -ne 0 ] && error "dumpToProducts failed"

UploadAsset
[ $? -ne 0 ] && error "UploadAsset failed"

exit 0


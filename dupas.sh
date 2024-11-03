#!/bin/bash


if [ -z "${1}" ] ; then
  echo "No input mask (first argument) was given." >&2
  exit 1
fi
iMask="${1}"

if [ -z "${2}" ] ; then
  echo "No output substitution (second argument) was given." >&2
  exit 1
fi
oMask="${2}"

rm -rf *${oMask}* runs/*${oMask}*
for item in *${iMask}* runs/*${iMask}*  ; do
  cp -r "$item" $(sed "s:$iMask:$oMask:g" <<< "$item")
done




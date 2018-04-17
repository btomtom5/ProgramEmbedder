#!/bin/bash

var1=dev_data

while getopts "cdf:" opt; do
  case $opt in
    c)
      echo "-c was triggered!" >&1
      ;;
    d)
       echo "-d was triggered!" >&1
       ;;
    f)
       echo "f was triggered!" >&1
       # echo $OPTARG
       echo $0
       shift $(($OPTIND - 2))
       echo $(dirname $0)
       var1="data"
       ;;

    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

if [ "$var1" == 'dev_data' ]; then
    echo "hello world"
fi

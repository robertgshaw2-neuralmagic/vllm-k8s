#!bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -m <MODEL_PATH> -b <BUCKET_URL>"
   exit 1 # Exit script after printing help
}

while getopts ":m:b:" opt;
do
   case "$opt" in
        m ) model_path="$OPTARG" ;;
        b ) bucket_url="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

function make_upload_cmd()
{
    cmd="gcloud storage cp"
    if [ ! -z $model_path ] ; then
        cmd+=" -r $model_path"
    else
        echo "Model name not provided"
        helpFunction
    fi

    if [ ! -z $bucket_url ] ; then
        cmd+=" $bucket_url/$model_path"
    else
        echo "Bucket url not provided"
        helpFunction
    fi
}

make_upload_cmd
echo ""
echo "Running: $cmd"
echo ""
$cmd
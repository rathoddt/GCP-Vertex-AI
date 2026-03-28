
# Custom Training with Pre built Container 
---

This lab demonsrates **Custom Training with Pre built Container** using scikit-learn

<!-- --- -->

## Step-01:  Create Vertex AI Workbench instance

Create Vertex AI instance Workbench in a region e.g. us-east1


Copy code in `sentiment_scikit_pre_trained` to cloud bucket

<!-- --- -->

## Step-02:  Mount bucket to Workbench
Mount bucket to Vertex AI workbench

Open terminal and change to `sentiment_scikit_pre_trained` directory where 
Run following command to package python files. 
```
python setup.py sdist --formats=gztar
```
That will create a folder and will create a file called `sentiment_analysis-1.0.tar.gz'. The file name and the version coming from setup.py


#### Move packaged code to package output directory
Move above .gz file to `package-output-directory` in a bucket location.


<!-- --- -->
## Step-03:  Create the training pipeleine
Create Vertex AI training pipeline
- Select pre-built container (TensorFlow 2.11)
- Specify package location in cloud storage tar file generate in previous step
- Python module: `trainer.task` as task.py is located here
- Select all other paratmeters as required
After training is finished (approximately in 10-12 mins) outputs will be stored at `gs://vertex-ai-rxperiments-01/01-custom-training-with-prebuilt-container/model-output-dir/sentiment_model`

## Step-04:  Importing and deploying the model
As this is custom training we need to manually import and model to **Model Registry** from the path given in previous step.

Once model is imported after while (approximately in 10-12 mins) it can be then deployed.

Deploy model to endpoint select parameter as required particularly machine type based on you budget. For learning, machine with 2 vcpu and 7.5 GB of memory is sufficient

## Step-05:  Testing model

Create `request.json` with following content 
```
{
  "instances": [
    "I love this movie",
    "It was terrible",
    "It was ok",
    "Best movie ever"
  ]
}
```

Set following variables

```
ENDPOINT_ID="<<>>"
PROJECT_ID="<<>>"
INPUT_DATA_FILE="path/to/your/input.json"
```

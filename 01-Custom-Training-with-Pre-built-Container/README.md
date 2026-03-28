
# Custom Training with Pre built Container 
---

This lab demonsrates **Custom Training with Pre built Container** using scikit-learn

---

## Step-01:  Create Vertex AI Workbench instance

Create Vertex AI instance Workbench in a region e.g. us-east1


Copy code in `sentiment_scikit_pre_trained` to cloud bucket

---

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


---
## Step-03:  Create the training pipeleine
Create Vertex AI training pipeline
- Select pre-built container (TensorFlow 2.11)
- Specify package location in cloud storage tar file generate in previous step
- Python module: `trainer.task` as task.py is located here
- Select all other paratmeters as required


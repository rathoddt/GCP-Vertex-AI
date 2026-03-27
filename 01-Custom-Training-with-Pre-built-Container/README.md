
# Custom Training with Pre built Container 
---

In this lab demonsrates **Custom Training with Pre built Container** using scikit-learn

---

## Step-01:  Create Vertex AI instance

### Step-01-01: Create VPC Security Group (for RDS)
Create Vertex AI instance in a region e.g. us-east1


Copy code in `sentiment_scikit_pre_trained` to cloud bucket

---

## Step-02:  Mount bucket to Workbench
Mount bucket to Vertex AI workbench

Open terminal and change to `sentiment_scikit_pre_trained` directory where 
Run following command to package python files. 
```
python setup.py sdist --formats=gztar
```
That will create a folder and will create a file called trainer-0.1.tar.gz. The file name and the version coming from setup.py


#### Move packaged code to package output directory
Move above .gz file to `package-output-directory` in a bucket location.


